# SPDX-License-Identifier: AGPL-3.0
"""Trainable kernel maximum-margin classification and residual BEV adapters.

This is an independently implemented experiment, not a reproduction of CBGS
or SSN. Inputs to the SVM are predicted geometry and decoder features, never
ground-truth geometry. Grayscale augmentation reduces, but cannot eliminate,
colour information in the learned decoder features.
"""
import math

import torch
from torch import nn
import torch.nn.functional as F


CLASS_NAMES = ('car', 'truck', 'construction_vehicle', 'bus', 'trailer',
               'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone')
# Car / large vehicles / barrier / two-wheelers / pedestrian / cone / background.
CLASS_GROUPS = (0, 1, 1, 1, 1, 2, 3, 3, 4, 5, 6)


class ShapeSVM(nn.Module):
    """Jointly trained multiclass SVM with a differentiable RBF approximation.

    Fine and coarse classifiers share their descriptor, not their final labels.
    Background has its own margin; fusion uses foreground-minus-background
    margins rather than interpreting SVM outputs as calibrated probabilities.

    The default RBF kernel is approximated by fixed random Fourier features
    (Rahimi-Recht). The learned linear max-margin heads and hinge objective
    remain fully differentiable into the descriptor, unlike a separately fit
    scikit-learn SVC. ``kernel='linear'`` preserves old checkpoint shapes.
    """
    def __init__(self, embed_dims, fusion_weight=0.1, loss_weight=0.1,
                 l2_weight=1e-4, negative_weight=0.1, max_negatives=128,
                 class_balance=False, kernel='rbf', rbf_features=512,
                 rbf_gamma=None):
        super().__init__()
        if kernel not in ('rbf', 'linear'):
            raise ValueError("SVM kernel must be 'rbf' or 'linear'")
        input_dims = embed_dims + 6
        if rbf_features < 1:
            raise ValueError('RBF feature count must be positive')
        if rbf_gamma is not None and rbf_gamma <= 0:
            raise ValueError('RBF gamma must be positive')
        self.kernel = kernel
        self.rbf_features = int(rbf_features)
        self.rbf_gamma = (1.0 / input_dims if rbf_gamma is None else float(rbf_gamma)) \
            if kernel == 'rbf' else None
        if kernel == 'rbf':
            # k(x,y)=exp(-gamma*||x-y||^2) ~= phi(x)^T phi(y).
            self.register_buffer('rff_omega', torch.randn(input_dims, self.rbf_features)
                                 * math.sqrt(2.0 * self.rbf_gamma))
            self.register_buffer('rff_phase', torch.rand(self.rbf_features) * (2.0 * math.pi))
            head_dims = self.rbf_features
        else:
            # None buffers are omitted from state_dict, preserving v2 linear checkpoints.
            self.register_buffer('rff_omega', None)
            self.register_buffer('rff_phase', None)
            head_dims = input_dims
        self.fine = nn.Linear(head_dims, 11)
        self.group = nn.Linear(head_dims, 7)
        self.register_buffer('class_groups', torch.tensor(CLASS_GROUPS))
        self.register_buffer('class_counts', torch.zeros(10, dtype=torch.long))
        self.register_buffer('fine_class_weights', torch.ones(11))
        self.register_buffer('group_class_weights', torch.ones(7))
        self.class_balance = class_balance
        self.fusion_weight = fusion_weight
        self.loss_weight = loss_weight
        self.l2_weight = l2_weight
        self.negative_weight = negative_weight
        self.max_negatives = max_negatives
        # Fusion initially leaves baseline scores exactly unchanged.
        for branch in (self.fine, self.group):
            nn.init.zeros_(branch.weight)
            nn.init.zeros_(branch.bias)

    @staticmethod
    def frequency_weights(counts):
        """Inverse square-root frequency, present-class mean 1, clipped [.25,4]."""
        counts = counts.float()
        present = counts > 0
        weights = torch.zeros_like(counts)
        if present.any():
            inverse = counts[present].rsqrt()
            weights[present] = (inverse / inverse.mean()).clamp(0.25, 4.0)
        return weights

    def set_class_counts(self, counts):
        counts = torch.as_tensor(counts, dtype=torch.long, device=self.class_counts.device)
        if counts.shape != (10,) or (counts < 0).any() or counts.sum() == 0:
            raise ValueError('Class balancing requires ten nonnegative training counts with some objects')
        self.class_counts.copy_(counts)
        if self.class_balance:
            self.fine_class_weights[:10].copy_(self.frequency_weights(counts))
            group_counts = torch.zeros(6, dtype=torch.long, device=counts.device)
            group_counts.scatter_add_(0, self.class_groups[:10], counts)
            self.group_class_weights[:6].copy_(self.frequency_weights(group_counts))

    def forward(self, query_features, boxes):
        # BEVFormer regression code: cx, cy, log(w), log(l), cz,
        # log(h), sin(yaw), cos(yaw), vx, vy.
        dims = torch.nan_to_num(boxes[..., [2, 3, 5]].float()).clamp(-5, 5)
        ratios = torch.stack((dims[..., 0] - dims[..., 1],
                              dims[..., 0] - dims[..., 2],
                              dims[..., 1] - dims[..., 2]), dim=-1)
        appearance = F.layer_norm(query_features.float(), (query_features.shape[-1],))
        descriptor = torch.cat((appearance, dims / 5, ratios / 10), dim=-1)
        if self.kernel == 'rbf':
            descriptor = self.rbf_map(descriptor)
        return (self.fine(descriptor.to(self.fine.weight.dtype)).float(),
                self.group(descriptor.to(self.group.weight.dtype)).float())

    def rbf_map(self, descriptor):
        """Map descriptors to a finite-dimensional approximation of an RBF kernel."""
        if self.kernel != 'rbf':
            raise ValueError('rbf_map is only available with kernel=rbf')
        projection = descriptor @ self.rff_omega.to(descriptor.dtype)
        projection = projection + self.rff_phase.to(descriptor.dtype)
        return math.sqrt(2.0 / self.rbf_features) * torch.cos(projection)

    def fuse(self, baseline_logits, fine, group):
        fine_margin = fine[..., :10] - fine[..., 10:11]
        group_margin = group[..., self.class_groups[:10]] - group[..., 6:7]
        correction = (fine_margin + 0.25 * group_margin).clamp(-5, 5)
        return baseline_logits + self.fusion_weight * correction.to(baseline_logits.dtype)

    @staticmethod
    def hinge(scores, labels):
        correct = scores.gather(1, labels[:, None])
        margins = (1 + scores - correct).clamp_min(0)
        margins = margins.scatter(1, labels[:, None], 0)
        return margins.sum(dim=1) / (scores.shape[1] - 1)

    def objective(self, fine, group, labels, label_weights, confidence):
        fine, group = fine.float().reshape(-1, 11), group.float().reshape(-1, 7)
        labels = labels.reshape(-1).long()
        valid = label_weights.reshape(-1) > 0
        positive = valid & (labels >= 0) & (labels < 10)
        negative = valid & (labels == 10)
        pos_inds, neg_inds = positive.nonzero().flatten(), negative.nonzero().flatten()
        if neg_inds.numel() > self.max_negatives:
            hardness = confidence.reshape(-1)[neg_inds].detach()
            neg_inds = neg_inds[hardness.topk(self.max_negatives).indices]
        # Keep gradients connected for empty scenes and distributed training.
        data_loss = (fine.sum() + group.sum()) * 0
        for indices, weight in ((pos_inds, 1.0), (neg_inds, self.negative_weight)):
            if indices.numel():
                targets = labels[indices]
                fine_loss = self.hinge(fine[indices], targets)
                coarse_loss = self.hinge(group[indices], self.class_groups[targets])
                if self.class_balance and weight == 1.0:
                    fine_loss = fine_loss * self.fine_class_weights[targets]
                    coarse_loss = coarse_loss * self.group_class_weights[self.class_groups[targets]]
                data_loss = data_loss + weight * (fine_loss + 0.25 * coarse_loss).mean()
        regularizer = 0.5 * (self.fine.weight.square().sum() + self.group.weight.square().sum())
        return self.loss_weight * (data_loss + self.l2_weight * regularizer)


class ResidualBottleneck(nn.Module):
    """C -> C/r -> C/r -> C; dimensions unchanged; exact identity at init."""
    def __init__(self, channels, ratio=4, time_emb_dims=None):
        super().__init__()
        if channels < 1 or ratio < 1:
            raise ValueError('Bottleneck channels and ratio must be positive')
        hidden = max(1, channels // ratio)
        groups = math.gcd(hidden, 32)
        self.reduce = nn.Conv2d(channels, hidden, 1)
        self.norm = nn.GroupNorm(groups, hidden)
        self.spatial = nn.Conv2d(hidden, hidden, 3, padding=1)
        self.expand = nn.Conv2d(hidden, channels, 1)
        self.time_gate = None
        if time_emb_dims is not None:
            self.time_gate = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dims, 1))
            nn.init.normal_(self.time_gate[-1].weight, std=1e-3)
            nn.init.zeros_(self.time_gate[-1].bias)
        nn.init.zeros_(self.expand.weight)
        nn.init.zeros_(self.expand.bias)

    def forward(self, x, embedding=None):
        residual = self.reduce(x)
        residual = F.silu(self.norm(residual))
        residual = F.silu(self.spatial(residual))
        residual = self.expand(residual)
        if self.time_gate is not None:
            if embedding is None:
                raise ValueError('Timestep-gated bottleneck requires the U-Net block embedding')
            # The existing embedding combines timestep and layout conditioning.
            gate = self.time_gate(embedding).sigmoid().to(residual.dtype)
            residual = residual * gate[:, :, None, None]
        return x + residual


def install_bottlenecks(unet, ratio=4, time_gate=False):
    """One adapter after each complete encoder/middle/decoder block.

    Hooks keep the original block names and pretrained parameter keys intact.
    Adapters are registered modules, so they participate in state_dict(),
    device conversion, optimizers and every U-Net call at train and test time.
    Skip features are saved after the adapter. No scheduler change is made.
    """
    if hasattr(unet, 'shape_adapters'):
        raise ValueError('Bottlenecks are already installed')
    adapters = nn.ModuleDict()
    blocks = [('encoder_{}'.format(i), b) for i, b in enumerate(unet.input_blocks)]
    blocks += [('middle', unet.middle_block)]
    blocks += [('decoder_{}'.format(i), b) for i, b in enumerate(unet.output_blocks)]
    for name, block in blocks:
        last = block[-1]
        channels = getattr(last, 'out_channels', getattr(last, 'channels', None))
        if channels is None:
            raise ValueError('Cannot determine output channels for {}'.format(name))
        time_emb_dims = unet.model_channels * 4 if time_gate else None
        adapters[name] = ResidualBottleneck(channels, ratio, time_emb_dims=time_emb_dims)
    unet.add_module('shape_adapters', adapters)
    for name, block in blocks:
        def adapt(module, inputs, output, key=name):
            h, extra = output
            return unet.shape_adapters[key](h, inputs[1]), extra
        block.register_forward_hook(adapt)
    return len(blocks)
