# Copyright (c) 2025
# SPDX-License-Identifier: AGPL-3.0
#
# U-ViT backbone for BEVDiffuser.
# Drop-in replacement for LayoutDiffusionUNetModel.
# Place this file at:
#   BEVFormer/projects/bevdiffuser/layout_diffusion/layout_uvit.py
#
# All values filled in from layout_tiny.py — no placeholders remain.

import torch
import torch.nn as nn

# ---- reuse existing modules exactly as layout_diffusion_unet.py does --------
from layout_diffusion.layout_encoder import LayoutTransformerEncoder
from layout_diffusion.layout_diffusion_unet import ObjectAwareCrossAttention
from layout_diffusion.nn import timestep_embedding


# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------

def modulate(x, shift, scale):
    """adaLN modulation. x:(B,N,C)  shift/scale:(B,C)"""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepMLP(nn.Module):
    """Sinusoidal timestep -> hidden vector, same idea as U-Net time_embed."""
    def __init__(self, hidden):
        super().__init__()
        self.hidden = hidden
        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )

    def forward(self, t):
        return self.mlp(timestep_embedding(t, self.hidden))


class PatchEmbed(nn.Module):
    """(B, C, H, W) -> (B, N, dim).
    patch=1 keeps every BEV cell as its own token — best for small objects."""
    def __init__(self, in_ch, dim, patch=1):
        super().__init__()
        self.patch = patch
        self.proj = nn.Conv2d(in_ch, dim, kernel_size=patch, stride=patch)

    def forward(self, x):
        x = self.proj(x)                      # (B, dim, H/p, W/p)
        B, D, Hh, Ww = x.shape
        x = x.flatten(2).transpose(1, 2)      # (B, N, dim)
        return x, (Hh, Ww)


class UnPatch(nn.Module):
    """(B, N, dim) -> (B, out_ch, H, W)."""
    def __init__(self, dim, out_ch, patch=1):
        super().__init__()
        self.patch = patch
        self.out_ch = out_ch
        self.proj = nn.Linear(dim, out_ch * patch * patch)

    def forward(self, x, grid):
        Hh, Ww = grid
        B, N, _ = x.shape
        p, c = self.patch, self.out_ch
        x = self.proj(x)                                       # (B, N, c*p*p)
        x = x.reshape(B, Hh, Ww, p, p, c)
        x = x.permute(0, 5, 1, 3, 2, 4).reshape(B, c, Hh*p, Ww*p)
        return x


# -----------------------------------------------------------------------------
# one U-ViT block:
#   self-attn (global, over BEV tokens)
#   -> object-aware cross-attn (layout, REUSED from LayoutDiffusionUNetModel)
#   -> MLP
# timestep enters via adaLN on self-attn and MLP sublayers (DiT-style)
# -----------------------------------------------------------------------------

class UViTBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()

        # --- self-attention ---
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn  = nn.MultiheadAttention(dim, num_heads, batch_first=True)

        # --- object-aware cross-attention (REUSED, values from layout_tiny.py) ---
        # resolution=12  comes from image_size(50) // ds(4) = 12
        # ds=4           is the first entry in attention_ds=[4,2,1]
        # encoder_channels=256  from hidden_dim=256 in layout_encoder config
        self.cross = ObjectAwareCrossAttention(
            channels=dim,
            num_heads=num_heads,
            encoder_channels=256,
            resolution=50,          # was 12
            ds=1,                   # was 4
            use_positional_embedding=True,
        )

        # --- MLP ---
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        hidden_mlp = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_mlp),
            nn.GELU(),
            nn.Linear(hidden_mlp, dim),
        )

        # --- adaLN: 6 vectors from timestep (shift+scale+gate for attn and mlp) ---
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)

    def forward(self, x, t_vec, cond_kwargs):
        """
        x          : (B, N, dim)   sequence of BEV tokens
        t_vec      : (B, dim)      timestep embedding
        cond_kwargs: dict          output of LayoutTransformerEncoder.forward()
        """
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = \
            self.ada(t_vec).chunk(6, dim=1)

        # self-attention (every BEV token sees every other)
        h = modulate(self.norm1(x), shift_a, scale_a)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + gate_a.unsqueeze(1) * h

        # object-aware cross-attention to layout
        # ObjectAwareCrossAttention expects (B, C, N) and returns (output, extra)
        B, N, C = x.shape
        xc = x.transpose(1, 2)                      # (B, C, N)
        xc, _ = self.cross(xc, cond_kwargs)         # returns tuple; [0] is (B,C,N)
        x = x + xc.transpose(1, 2)                  # back to (B, N, C)

        # MLP
        h = modulate(self.norm2(x), shift_m, scale_m)
        x = x + gate_m.unsqueeze(1) * self.mlp(h)

        return x


# -----------------------------------------------------------------------------
# U-ViT backbone  (long skip connections between encoder and decoder halves)
# -----------------------------------------------------------------------------

class LayoutDiTModel(nn.Module):
    """
    U-ViT-style diffusion backbone for BEVDiffuser.

    Public contract (same as LayoutDiffusionUNetModel):
      - built by build_unet(cfg.unet) with type='LayoutDiTModel'
      - forward(x, timesteps, **cond) -> (pred,)   pred.shape == x.shape
      - exposes .layout_encoder  (train loop reads attributes on it)
      - exposes .downsample_blocks / .upsample_blocks  (for checkpoint branch)
    """

    def __init__(
        self,
        image_size=50,
        in_channels=256,
        out_channels=256,
        dim=512,
        depth=12,           # must be even: first half = encoder, second = decoder
        num_heads=8,
        patch=1,            # patch=1 keeps every BEV cell; best for small objects
        mlp_ratio=4.0,
        layout_encoder=None,
        **kwargs,           # absorb any extra config keys silently
    ):
        super().__init__()
        assert depth % 2 == 0, "depth must be even for symmetric long skips"

        self.image_size   = image_size
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.patch        = patch

        # ---- layout encoder (REUSED — same config as layout_tiny.py) --------
        # layout_encoder dict arrives from the config, with a 'parameters' sub-dict
        le_params = layout_encoder["parameters"] if "parameters" in layout_encoder \
                    else layout_encoder
        self.layout_encoder = LayoutTransformerEncoder(**le_params)

        # ---- timestep embedding ---------------------------------------------
        self.t_embed = TimestepMLP(dim)

        # ---- patch embedding ------------------------------------------------
        self.patch_embed = PatchEmbed(in_channels, dim, patch)
        n_tokens = (image_size // patch) ** 2
        self.pos = nn.Parameter(torch.zeros(1, n_tokens, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)

        # ---- transformer blocks ---------------------------------------------
        self.blocks = nn.ModuleList([
            UViTBlock(dim, num_heads, mlp_ratio)
            for _ in range(depth)
        ])

        # ---- long skip fusion (U-ViT signature) -----------------------------
        # Each decoder block receives [h_decoder || h_skip] -> linear -> dim
        self.skip_fuse = nn.ModuleList([
            nn.Linear(2 * dim, dim)
            for _ in range(depth // 2)
        ])

        # ---- output head ----------------------------------------------------
        self.norm_out = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.unpatch  = UnPatch(dim, out_channels, patch)

        # keep these attributes so the pretrained-checkpoint freeze branch works
        # (train.sh checks for .downsample_blocks / .upsample_blocks)
        self.downsample_blocks = nn.ModuleList()
        self.upsample_blocks   = nn.ModuleList()

    def forward(self, x, timesteps, **cond):
        """
        x         : (B, in_channels, H, W)   noisy BEV feature map
        timesteps : (B,)                      diffusion timestep
        **cond    : obj_class, obj_bbox, is_valid_obj  (from get_condition)

        returns   : (pred,)   pred.shape == x.shape
        """

        # 1. layout encoder -> cond_kwargs dict
        #    keys used by ObjectAwareCrossAttention:
        #      xf_proj, xf_out, key_padding_mask,
        #      image_patch_bbox_embedding_for_resolution12  (resolution=12)
        cond_kwargs = self.layout_encoder(**cond)

        # 2. timestep embedding
        t_vec = self.t_embed(timesteps)          # (B, dim)

        # 3. patchify + positional embedding
        h, grid = self.patch_embed(x)            # (B, N, dim)
        h = h + self.pos

        # 4. U-ViT forward: encoder half saves skips, decoder half fuses them
        half   = len(self.blocks) // 2
        skips  = []

        for i, blk in enumerate(self.blocks):
            if i < half:
                # encoder: run block, save output for the matching decoder block
                h = blk(h, t_vec, cond_kwargs)
                skips.append(h)
            else:
                # decoder: fuse with the mirror encoder output (long skip)
                s = skips.pop()                              # LIFO -> mirror match
                h = self.skip_fuse[i - half](
                    torch.cat([h, s], dim=-1)               # (B, N, 2*dim) -> (B, N, dim)
                )
                h = blk(h, t_vec, cond_kwargs)

        # 5. output head -> BEV grid
        h    = self.norm_out(h)
        pred = self.unpatch(h, grid)             # (B, out_channels, H, W)

        return (pred,)


# -----------------------------------------------------------------------------
# Isotropic DiT variant — same blocks, NO long skip connections.
# The comparison arm: lets you test "do skips matter for long-tail BEV objects?"
# Switch between the two by changing unet.type in the config — nothing else changes.
# -----------------------------------------------------------------------------

class LayoutDiTModelIsotropic(LayoutDiTModel):
    """
    Plain DiT: identical to LayoutDiTModel but without the long skip connections.
    Use type='LayoutDiTModelIsotropic' in layout_tiny_dit.py config.
    """

    def forward(self, x, timesteps, **cond):
        cond_kwargs = self.layout_encoder(**cond)
        t_vec       = self.t_embed(timesteps)
        h, grid     = self.patch_embed(x)
        h           = h + self.pos

        # straight stack — no skips, no skip_fuse
        for blk in self.blocks:
            h = blk(h, t_vec, cond_kwargs)

        pred = self.unpatch(self.norm_out(h), grid)
        return (pred,)
