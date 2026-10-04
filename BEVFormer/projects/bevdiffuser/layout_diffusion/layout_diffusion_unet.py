# Copyright (c) 2025 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

# This source code is derived from LayoutDiffusion
#   (https://github.com/ZGCTroy/LayoutDiffusion)
# Copyright (c) 2023 LayoutDiffusion authors, licensed under the MIT license,
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.
#
# MODIFIED: Added Option 3 — Layout-Aware Pre-Conditioning (fine, resolution=50)
# Three edits marked with ### OPTION 3 ###

from abc import abstractmethod
import os
import safetensors
import math
import numpy as np
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .nn import (
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)
from diffusers.utils.constants import SAFETENSORS_WEIGHTS_NAME

def convert_module_to_f16(l):
    if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        l.weight.data = l.weight.data.half()
        if l.bias is not None:
            l.bias.data = l.bias.data.half()

class SiLU(nn.Module):
    @staticmethod
    def forward(x):
        return x * th.sigmoid(x)

class TimestepBlock(nn.Module):
    @abstractmethod
    def forward(self, x, emb):
        pass

class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    def forward(self, x, emb, cond_kwargs=None):
        extra_output = None
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, (AttentionBlock, ObjectAwareCrossAttention)):
                x, extra_output = layer(x, cond_kwargs)
            else:
                x = layer(x)
        return x, extra_output

class Upsample(nn.Module):
    def __init__(self, channels, use_conv, dims=2, out_channels=None, out_size=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.out_size = out_size
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            if self.out_size is None:
                x = F.interpolate(x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest")
            else:
                x = F.interpolate(x, (x.shape[2], self.out_size, self.out_size), mode="nearest")
        else:
            if self.out_size is None:
                x = F.interpolate(x, scale_factor=2, mode="nearest")
            else:
                x = F.interpolate(x, size=self.out_size, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x

class Downsample(nn.Module):
    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(dims, self.channels, self.out_channels, 3, stride=stride, padding=1)
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)

class ResBlock(TimestepBlock):
    def __init__(self, channels, emb_channels, dropout, out_channels=None, use_conv=False,
                 use_scale_shift_norm=False, dims=2, use_checkpoint=False, up=False, down=False, out_size=None):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm
        self.in_layers = nn.Sequential(normalization(channels), SiLU(), conv_nd(dims, channels, self.out_channels, 3, padding=1))
        self.updown = up or down
        if up:
            self.h_upd = Upsample(channels, False, dims, out_size=out_size)
            self.x_upd = Upsample(channels, False, dims, out_size=out_size)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()
        self.emb_layers = nn.Sequential(SiLU(), linear(emb_channels, 2 * self.out_channels if use_scale_shift_norm else self.out_channels))
        self.out_layers = nn.Sequential(normalization(self.out_channels), SiLU(), nn.Dropout(p=dropout), zero_module(conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)))
        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 3, padding=1)
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h

class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=1, num_head_channels=-1, use_checkpoint=False,
                 encoder_channels=None, return_attention_embeddings=False, ds=None, resolution=None,
                 type=None, use_positional_embedding=False, **kwargs):
        super().__init__()
        self.type = type
        self.ds = ds
        self.resolution = resolution
        self.return_attention_embeddings = return_attention_embeddings
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert channels % num_head_channels == 0
            self.num_heads = channels // num_head_channels
        self.use_positional_embedding = use_positional_embedding
        if self.use_positional_embedding:
            self.positional_embedding = nn.Parameter(th.randn(channels // self.num_heads, resolution ** 2) / channels ** 0.5)
        else:
            self.positional_embedding = None
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        self.attention = QKVAttentionLegacy(self.num_heads)
        self.encoder_channels = encoder_channels
        if encoder_channels is not None:
            self.encoder_kv = conv_nd(1, encoder_channels, channels * 2, 1)
        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x, cond_kwargs=None):
        extra_output = None
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        if cond_kwargs is not None and self.encoder_channels is not None:
            kv_for_encoder_out = self.encoder_kv(cond_kwargs['xf_out'])
            h = self.attention(qkv, kv_for_encoder_out, positional_embedding=self.positional_embedding)
        else:
            h = self.attention(qkv, positional_embedding=self.positional_embedding)
        h = self.proj_out(h)
        output = (x + h).reshape(b, c, *spatial)
        if self.return_attention_embeddings:
            assert cond_kwargs is not None
            if extra_output is None:
                extra_output = {}
            extra_output.update({'type': self.type, 'ds': self.ds, 'resolution': self.resolution,
                                 'num_heads': self.num_heads, 'num_channels': self.channels,
                                 'image_query_embeddings': qkv[:, :self.channels, :].detach()})
            if cond_kwargs is not None:
                extra_output.update({'layout_key_embeddings': kv_for_encoder_out[:, :self.channels, :].detach()})
        return output, extra_output

class ObjectAwareCrossAttention(nn.Module):
    def __init__(self, channels, num_heads=1, num_head_channels=-1, use_checkpoint=False,
                 encoder_channels=None, return_attention_embeddings=False, ds=None, resolution=None,
                 type=None, use_positional_embedding=True, use_key_padding_mask=False,
                 channels_scale_for_positional_embedding=1.0, norm_first=False, norm_for_obj_embedding=False):
        super().__init__()
        self.norm_for_obj_embedding = None
        self.norm_first = norm_first
        self.channels_scale_for_positional_embedding = channels_scale_for_positional_embedding
        self.use_key_padding_mask = use_key_padding_mask
        self.type = type
        self.ds = ds
        self.resolution = resolution
        self.return_attention_embeddings = return_attention_embeddings
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert channels % num_head_channels == 0
            self.num_heads = channels // num_head_channels
        self.use_positional_embedding = use_positional_embedding
        assert self.use_positional_embedding
        self.use_checkpoint = use_checkpoint
        self.qkv_projector = conv_nd(1, channels, 3 * channels, 1)
        self.norm_for_qkv = normalization(channels)
        if encoder_channels is not None:
            self.encoder_channels = encoder_channels
            self.layout_content_embedding_projector = conv_nd(1, encoder_channels, channels * 2, 1)
            self.layout_position_embedding_projector = conv_nd(1, encoder_channels, int(channels * self.channels_scale_for_positional_embedding), 1)
            if self.norm_first:
                if norm_for_obj_embedding:
                    self.norm_for_obj_embedding = normalization(encoder_channels)
                self.norm_for_obj_class_embedding = normalization(encoder_channels)
                self.norm_for_layout_positional_embedding = normalization(encoder_channels)
                self.norm_for_image_patch_positional_embedding = normalization(encoder_channels)
            else:
                self.norm_for_obj_class_embedding = normalization(encoder_channels)
                self.norm_for_layout_positional_embedding = normalization(int(channels * self.channels_scale_for_positional_embedding))
                self.norm_for_image_patch_positional_embedding = normalization(int(channels * self.channels_scale_for_positional_embedding))
        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x, cond_kwargs):
        extra_output = None
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv_projector(self.norm_for_qkv(x))
        bs, C, L1, L2 = qkv.shape[0], self.channels, qkv.shape[2], cond_kwargs['obj_bbox_embedding'].shape[-1]
        if self.norm_first:
            image_patch_positional_embedding = self.norm_for_image_patch_positional_embedding(cond_kwargs['image_patch_bbox_embedding_for_resolution{}'.format(self.resolution)])
            image_patch_positional_embedding = self.layout_position_embedding_projector(image_patch_positional_embedding)
        else:
            image_patch_positional_embedding = self.layout_position_embedding_projector(cond_kwargs['image_patch_bbox_embedding_for_resolution{}'.format(self.resolution)])
            image_patch_positional_embedding = self.norm_for_image_patch_positional_embedding(image_patch_positional_embedding)
        image_patch_positional_embedding = image_patch_positional_embedding.reshape(bs * self.num_heads, int(C * self.channels_scale_for_positional_embedding) // self.num_heads, L1)
        q_image_patch_content_embedding, k_image_patch_content_embedding, v_image_patch_content_embedding = qkv.split(C, dim=1)
        q_image_patch_content_embedding = q_image_patch_content_embedding.reshape(bs * self.num_heads, C // self.num_heads, L1)
        k_image_patch_content_embedding = k_image_patch_content_embedding.reshape(bs * self.num_heads, C // self.num_heads, L1)
        v_image_patch_content_embedding = v_image_patch_content_embedding.reshape(bs * self.num_heads, C // self.num_heads, L1)
        q_image_patch = torch.cat([q_image_patch_content_embedding, image_patch_positional_embedding], dim=1)
        k_image_patch = torch.cat([k_image_patch_content_embedding, image_patch_positional_embedding], dim=1)
        v_image_patch = v_image_patch_content_embedding
        if self.norm_first:
            layout_positional_embedding = self.norm_for_layout_positional_embedding(cond_kwargs['obj_bbox_embedding'])
            layout_positional_embedding = self.layout_position_embedding_projector(layout_positional_embedding)
        else:
            layout_positional_embedding = self.layout_position_embedding_projector(cond_kwargs['obj_bbox_embedding'])
            layout_positional_embedding = self.norm_for_layout_positional_embedding(layout_positional_embedding)
        layout_positional_embedding = layout_positional_embedding.reshape(bs * self.num_heads, int(C * self.channels_scale_for_positional_embedding) // self.num_heads, L2)
        if self.norm_for_obj_embedding is not None:
            layout_content_embedding = (self.norm_for_obj_embedding(cond_kwargs['xf_out']) + self.norm_for_obj_class_embedding(cond_kwargs['obj_class_embedding'])) / 2
        else:
            layout_content_embedding = (cond_kwargs['xf_out'] + self.norm_for_obj_class_embedding(cond_kwargs['obj_class_embedding'])) / 2
        k_layout_content_embedding, v_layout_content_embedding = self.layout_content_embedding_projector(layout_content_embedding).split(C, dim=1)
        k_layout_content_embedding = k_layout_content_embedding.reshape(bs * self.num_heads, C // self.num_heads, L2)
        v_layout_content_embedding = v_layout_content_embedding.reshape(bs * self.num_heads, C // self.num_heads, L2)
        k_layout = torch.cat([k_layout_content_embedding, layout_positional_embedding], dim=1)
        v_layout = v_layout_content_embedding
        k_mix = th.cat([k_image_patch, k_layout], dim=2)
        v_mix = th.cat([v_image_patch, v_layout], dim=2)
        if self.use_key_padding_mask:
            key_padding_mask = torch.cat([torch.zeros((bs, L1), device=cond_kwargs['key_padding_mask'].device).bool(), cond_kwargs['key_padding_mask']], dim=1)
            print(cond_kwargs['key_padding_mask'])
        scale = 1 / math.sqrt(math.sqrt(int((1 + self.channels_scale_for_positional_embedding) * C) // self.num_heads))
        attn_output_weights = th.einsum("bct,bcs->bts", q_image_patch * scale, k_mix * scale)
        attn_output_weights = attn_output_weights.view(bs, self.num_heads, L1, L1 + L2)
        if self.use_key_padding_mask:
            attn_output_weights = attn_output_weights.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        attn_output_weights = attn_output_weights.view(bs * self.num_heads, L1, L1 + L2)
        attn_output_weights = th.softmax(attn_output_weights.float(), dim=-1).type(attn_output_weights.dtype)
        attn_output = th.einsum("bts,bcs->bct", attn_output_weights, v_mix)
        attn_output = attn_output.reshape(bs, C, L1)
        h = self.proj_out(attn_output)
        output = (x + h).reshape(b, c, *spatial)
        if self.return_attention_embeddings:
            assert cond_kwargs is not None
            if extra_output is None:
                extra_output = {}
            extra_output.update({'type': self.type, 'ds': self.ds, 'resolution': self.resolution,
                                 'num_heads': self.num_heads, 'num_channels': self.channels,
                                 'image_query_embeddings': image_patch_positional_embedding.detach().view(bs, -1, L1)})
            if cond_kwargs is not None:
                extra_output.update({'layout_key_embeddings': layout_positional_embedding.detach().view(bs, -1, L2)})
        return output, extra_output

def count_flops_attn(model, _x, y):
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    matmul_ops = 2 * b * (num_spatial ** 2) * c
    model.total_ops += th.DoubleTensor([matmul_ops])

class QKVAttentionLegacy(nn.Module):
    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv, encoder_kv=None, positional_embedding=None):
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        if positional_embedding is not None:
            q = q + positional_embedding[None, :, :].to(q.dtype)
            k = k + positional_embedding[None, :, :].to(q.dtype)
        if encoder_kv is not None:
            assert encoder_kv.shape[1] == self.n_heads * ch * 2
            ek, ev = encoder_kv.reshape(bs * self.n_heads, ch * 2, -1).split(ch, dim=1)
            k = th.cat([ek, k], dim=-1)
            v = th.cat([ev, v], dim=-1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum("bct,bcs->bts", q * scale, k * scale)
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class LayoutDiffusionUNetModel(nn.Module):
    """
    A UNetModel that conditions on layout with an encoding transformer.
    
    MODIFIED: Added Option 3 — Layout-Aware Pre-Conditioning.
    A single ObjectAwareCrossAttention at FULL BEV resolution (50x50)
    runs BEFORE the input_blocks, giving the denoiser a spatially-aware
    starting point. Controlled by use_preconditioning flag.
    """

    def __init__(
            self,
            layout_encoder,
            in_channels,
            model_channels,
            out_channels,
            num_res_blocks,
            attention_ds,
            encoder_channels=None,
            dropout=0,
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            dims=2,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            resblock_updown=False,
            use_positional_embedding_for_attention=False,
            image_size=256,
            attention_block_type='GLIDE',
            num_attention_blocks=1,
            use_key_padding_mask=False,
            channels_scale_for_positional_embedding=1.0,
            norm_first=False,
            norm_for_obj_embedding=False,
            num_pre_downsample=0,
            use_preconditioning=False,       ### OPTION 3 ### new parameter
    ):
        super().__init__()
        self.norm_for_obj_embedding = norm_for_obj_embedding
        self.channels_scale_for_positional_embedding = channels_scale_for_positional_embedding
        self.norm_first = norm_first
        self.use_key_padding_mask = use_key_padding_mask
        self.num_attention_blocks = num_attention_blocks
        self.attention_block_type = attention_block_type
        if self.attention_block_type == 'GLIDE':
            attention_block_fn = AttentionBlock
        elif self.attention_block_type == 'ObjectAwareCrossAttention':
            attention_block_fn = ObjectAwareCrossAttention

        self.image_size = image_size
        self.use_positional_embedding_for_attention = use_positional_embedding_for_attention
        self.layout_encoder = layout_encoder

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.in_channels = in_channels
        self.encoder_channels = encoder_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_ds = attention_ds
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )
        self.downsample_blocks = nn.ModuleList([])
        self.upsample_blocks = nn.ModuleList([])
        for _ in range(num_pre_downsample):
            self.downsample_blocks.append(Downsample(in_channels, conv_resample, dims=dims, out_channels=in_channels))
            self.upsample_blocks.append(Upsample(out_channels, conv_resample, dims=dims, out_channels=out_channels))
            self.image_size = self.image_size // 2

        ch = input_ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))]
        )
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [ResBlock(ch, time_embed_dim, dropout, out_channels=int(mult * model_channels),
                                   dims=dims, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm)]
                ch = int(mult * model_channels)
                if ds in attention_ds:
                    print('encoder attention layer: ds = {}, resolution = {}'.format(ds, self.image_size // ds))
                    for _ in range(self.num_attention_blocks):
                        layers.append(attention_block_fn(ch, use_checkpoint=use_checkpoint, num_heads=num_heads,
                            num_head_channels=num_head_channels, encoder_channels=encoder_channels, ds=ds,
                            resolution=int(self.image_size // ds), type='input',
                            use_positional_embedding=self.use_positional_embedding_for_attention,
                            use_key_padding_mask=self.use_key_padding_mask,
                            channels_scale_for_positional_embedding=self.channels_scale_for_positional_embedding,
                            norm_first=self.norm_first, norm_for_obj_embedding=self.norm_for_obj_embedding))
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(ch, time_embed_dim, dropout, out_channels=out_ch, dims=dims,
                                 use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm, down=True)
                        if resblock_updown
                        else Downsample(ch, conv_resample, dims=dims, out_channels=out_ch)))
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        print('middle attention layer: ds = {}, resolution = {}'.format(ds, self.image_size // ds))
        self.middle_block = TimestepEmbedSequential(
            ResBlock(ch, time_embed_dim, dropout, dims=dims, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm),
            attention_block_fn(ch, use_checkpoint=use_checkpoint, num_heads=num_heads, num_head_channels=num_head_channels,
                encoder_channels=encoder_channels, ds=ds, resolution=int(self.image_size // ds), type='middle',
                use_positional_embedding=self.use_positional_embedding_for_attention,
                use_key_padding_mask=self.use_key_padding_mask,
                channels_scale_for_positional_embedding=self.channels_scale_for_positional_embedding,
                norm_first=self.norm_first, norm_for_obj_embedding=self.norm_for_obj_embedding),
            ResBlock(ch, time_embed_dim, dropout, dims=dims, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [ResBlock(ch + ich, time_embed_dim, dropout, out_channels=int(model_channels * mult),
                                   dims=dims, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm)]
                ch = int(model_channels * mult)
                if ds in attention_ds:
                    print('decoder attention layer: ds = {}, resolution = {}'.format(ds, self.image_size // ds))
                    for _ in range(self.num_attention_blocks):
                        layers.append(attention_block_fn(ch, use_checkpoint=use_checkpoint, num_heads=num_heads_upsample,
                            num_head_channels=num_head_channels, encoder_channels=encoder_channels, ds=ds,
                            resolution=int(self.image_size // ds), type='output',
                            use_positional_embedding=self.use_positional_embedding_for_attention,
                            use_key_padding_mask=self.use_key_padding_mask,
                            channels_scale_for_positional_embedding=self.channels_scale_for_positional_embedding,
                            norm_first=self.norm_first, norm_for_obj_embedding=self.norm_for_obj_embedding))
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(ch, time_embed_dim, dropout, out_channels=out_ch, dims=dims,
                                 use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm,
                                 up=True, out_size=int(self.image_size // (ds // 2)))
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch, out_size=int(self.image_size // ds)))
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(normalization(ch), SiLU(), zero_module(conv_nd(dims, input_ch, out_channels, 3, padding=1)))
        self.use_fp16 = use_fp16

        ### OPTION 3 — EDIT 1: build pre-conditioning cross-attention ###########
        self.use_preconditioning = use_preconditioning
        if self.use_preconditioning:
            # FINE version: operates at FULL BEV resolution (image_size x image_size)
            # For tiny: 50x50 = 2500 tokens. Uses resolution_to_attention=[12,25,50]
            # so the layout encoder already emits the key for resolution=image_size.
            self.precond_resolution = self.image_size   # 50 for tiny
            self.precondition_attn = ObjectAwareCrossAttention(
                channels=in_channels,                   # 256 — matches h before input_blocks
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                encoder_channels=encoder_channels,      # 256
                resolution=self.precond_resolution,     # 50
                ds=1,
                type='precond',
                use_positional_embedding=True,
                use_key_padding_mask=use_key_padding_mask,
                channels_scale_for_positional_embedding=channels_scale_for_positional_embedding,
                norm_first=norm_first,
                norm_for_obj_embedding=norm_for_obj_embedding,
            )
            # zero-init gate: starts as identity (no effect), grows only if helpful
            # same trick as GLIGEN/ControlNet — prevents destabilizing early training
            self.precond_gate = nn.Parameter(th.zeros(1))
            print('Option 3: pre-conditioning cross-attention at resolution={}'.format(self.precond_resolution))
        #########################################################################

    ### OPTION 3 — EDIT 2: helper method ########################################
    def apply_preconditioning(self, h, layout_outputs):
        """
        Run one cross-attention pass where BEV cells attend to layout objects
        BEFORE the U-Net's down blocks. Gives the denoiser a spatially-aware
        starting point.

        h              : (B, C, H, W)  BEV features after downsample_blocks
        layout_outputs : dict from layout encoder (xf_out, obj_bbox_embedding, etc.)
        returns        : (B, C, H, W)  pre-conditioned BEV
        """
        if not self.use_preconditioning:
            return h

        B, C, H, W = h.shape
        r = self.precond_resolution   # 50 for tiny

        # If h is already at the target resolution, no interpolation needed.
        # If h was downsampled by pre-downsample blocks, interpolate.
        if H != r or W != r:
            h_for_attn = F.interpolate(h, size=(r, r), mode='bilinear', align_corners=False)
        else:
            h_for_attn = h

        # ObjectAwareCrossAttention expects (B, C, L) and returns (output, extra)
        h_seq = h_for_attn.reshape(B, C, r * r)            # (B, C, 2500)
        attn_out, _ = self.precondition_attn(h_seq, layout_outputs)  # (B, C, 2500)

        # reshape back to spatial
        attn_out = attn_out.reshape(B, C, r, r)             # (B, C, 50, 50)

        # upsample back if we downsampled earlier
        if H != r or W != r:
            attn_out = F.interpolate(attn_out, size=(H, W), mode='bilinear', align_corners=False)

        # gated residual: gate starts at 0, so initial behavior is identity
        return h + self.precond_gate * attn_out
    #############################################################################

    def convert_to_fp16(self):
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)
        self.layout_encoder.convert_to_fp16()

    def forward(self, x, timesteps, obj_class=None, obj_bbox=None, obj_mask=None, is_valid_obj=None, obj_name=None,obj_time=None, **kwargs):
        hs, extra_outputs = [], []

        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        layout_outputs = self.layout_encoder(
            obj_class=obj_class,
            obj_bbox=obj_bbox,
            obj_mask=obj_mask,
            is_valid_obj=is_valid_obj,
            obj_name=obj_name,
            obj_time=obj_time
        )
        xf_proj, xf_out = layout_outputs["xf_proj"], layout_outputs["xf_out"]

        emb = emb + xf_proj.to(emb)

        h = x.type(self.dtype)
        for module in self.downsample_blocks:
            h = module(h)

        ### OPTION 3 — EDIT 3: apply pre-conditioning before input_blocks #######
        h = self.apply_preconditioning(h, layout_outputs)
        #########################################################################

        for module in self.input_blocks:
            h, extra_output = module(h, emb, layout_outputs)
            if extra_output is not None:
                extra_outputs.append(extra_output)
            hs.append(h)
        h, extra_output = self.middle_block(h, emb, layout_outputs)
        if extra_output is not None:
            extra_outputs.append(extra_output)
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h, extra_output = module(h, emb, layout_outputs)
            if extra_output is not None:
                extra_outputs.append(extra_output)
        h = h.type(x.dtype)
        h = self.out(h)
        for module in self.upsample_blocks:
            h = module(h)

        return [h, extra_outputs]

    def save_pretrained(self, save_directory):
        if os.path.isfile(save_directory):
            print(f"Provided path ({save_directory}) should be a directory, not a file")
            return
        os.makedirs(save_directory, exist_ok=True)
        weights_name = SAFETENSORS_WEIGHTS_NAME
        safetensors.torch.save_file(self.state_dict(), os.path.join(save_directory, weights_name), metadata={"format": "pt"})

    def from_pretrained(self, pretrained_model_name_or_path, subfolder=None):
        weights_name = SAFETENSORS_WEIGHTS_NAME
        if os.path.isfile(pretrained_model_name_or_path):
            checkpoint_file = pretrained_model_name_or_path
        elif os.path.isdir(pretrained_model_name_or_path):
            if os.path.isfile(os.path.join(pretrained_model_name_or_path, weights_name)):
                checkpoint_file = os.path.join(pretrained_model_name_or_path, weights_name)
            elif subfolder is not None and os.path.isfile(os.path.join(pretrained_model_name_or_path, subfolder, weights_name)):
                checkpoint_file = os.path.join(pretrained_model_name_or_path, subfolder, weights_name)
        else:
            print(f"Error no file named {weights_name} found in directory {pretrained_model_name_or_path}.")
            return
        state_dict = safetensors.torch.load_file(checkpoint_file, device="cpu")
        try:
            self.load_state_dict(state_dict, strict=True)
            print('successfully load the entire model')
        except:
            print('not successfully load the entire model, try to load part of model')
            self.load_state_dict(state_dict, strict=False)