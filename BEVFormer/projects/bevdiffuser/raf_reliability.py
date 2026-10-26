# SPDX-License-Identifier: AGPL-3.0
"""Camera-only RAF-inspired reliability gating, independently implemented.

Reference: parkie0517/RAF, commit a64c2607 (ConfHead and minimum-weight
feature gating). No reference source is vendored. Original RAF uses learned
LiDAR/image similarity, cleanliness labels, and CALM to supervise reliability.
Here known synthetic corruption strength supervises *augmentation integrity*,
not calibrated detection confidence or the original cross-modal RAF target.
"""
import copy
import json
import math
import os
from contextlib import contextmanager

import torch
from torch import nn
import torch.nn.functional as F


class CameraReliability(nn.Module):
    """Shared head across cameras and FPN levels, with no camera reordering."""
    def __init__(self, channels=256, hidden_channels=128, min_weight=0.2):
        super().__init__()
        if channels <= 0 or hidden_channels <= 0 or not 0 <= min_weight < 1:
            raise ValueError("Invalid RAF channels or minimum weight")
        self.config = dict(channels=channels, hidden_channels=hidden_channels,
                           min_weight=min_weight, supervision="synthetic_integrity_v1",
                           version=1)
        self.head = nn.Sequential(nn.Conv2d(channels, hidden_channels, 3, padding=1),
                                  nn.ReLU(), nn.Conv2d(hidden_channels, 1, 1))
        # Near-identity initial gate, with a nonzero final kernel so all layers
        # can receive gradients on the first training step.
        nn.init.normal_(self.head[-1].weight, std=1e-3)
        nn.init.constant_(self.head[-1].bias, math.log(0.95 / 0.05))

    def forward(self, features, integrity=None):
        if not isinstance(features, (tuple, list)) or not features:
            raise ValueError("RAF expects a nonempty FPN feature pyramid")
        gated, maps, losses = [], [], []
        for feature in features:
            if feature.ndim != 4 or feature.shape[1] != self.config['channels']:
                raise ValueError("RAF expects flattened [B*N,C,H,W] FPN levels")
            logits = self.head(feature.float())
            reliability = logits.sigmoid()
            weight = self.config['min_weight'] + (1-self.config['min_weight']) * reliability
            gated.append(feature * weight.to(feature.dtype))
            maps.append(reliability)
            if integrity is not None:
                if integrity.shape[0] != feature.shape[0]:
                    raise ValueError("Camera/image target batch ordering does not match FPN")
                target = F.interpolate(integrity.float(), size=feature.shape[-2:], mode='area')
                # Balanced clean/corrupt regions discourage constant maps even
                # when the unaffected background occupies most of an image.
                bce = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
                corrupt = (target < 0.999).float()
                clean = 1-corrupt
                terms = []
                for mask in (clean, corrupt):
                    if mask.sum() > 0:
                        terms.append((bce*mask).sum()/mask.sum())
                losses.append(torch.stack(terms).mean())
        loss = torch.stack(losses).mean() if losses else maps[0].sum()*0
        return tuple(gated), loss, tuple(maps)


def corrupt_normalized_images(images):
    """Known synthetic occlusion/blur/noise; neither weather simulation nor restoration.

    Operates after existing normalization/GridMask without altering image
    dimensions or calibration. Half of cameras remain unchanged in expectation.
    The target 1-blend_strength describes added corruption, not real-world truth.
    """
    if images.ndim != 4 or not images.is_floating_point():
        raise ValueError("RAF image backbone input must be floating [B*N,3,H,W]")
    n, _, h, w = images.shape
    corruption = torch.zeros(n, 1, h, w, device=images.device, dtype=images.dtype)
    # Independent sampling per flattened camera; ordering is preserved.
    for i in range(n):
        if torch.rand((), device=images.device).item() < 0.5:
            continue
        rh = max(1, int(h * (0.2+0.6*torch.rand((), device=images.device).item())))
        rw = max(1, int(w * (0.2+0.6*torch.rand((), device=images.device).item())))
        y = int(torch.randint(h-rh+1, (), device=images.device).item())
        x = int(torch.randint(w-rw+1, (), device=images.device).item())
        strength = 0.4+0.6*torch.rand((), device=images.device).item()
        corruption[i, :, y:y+rh, x:x+rw] = strength
    blurred = F.avg_pool2d(images, 9, stride=1, padding=4)
    scale = images.float().std(dim=(-2, -1), keepdim=True, unbiased=False).clamp_min(1e-3)
    noise = torch.randn_like(images)*scale.to(images.dtype)
    # Choose blur/noise or opaque lens-like occlusion independently per view.
    opaque = (torch.rand(n, 1, 1, 1, device=images.device) < 0.5).to(images.dtype)
    degraded = (1-opaque)*(blurred+noise)
    result = images*(1-corruption)+degraded*corruption
    return result, (1-corruption).detach()


class ReliabilityController:
    """Scoped hooks on frozen image backbone/FPN; absent when RAF is disabled.

    Only the auxiliary reliability head executes with gradients; the encoder
    remains under no_grad. Diffusion/task gradients train the existing U-Net
    and detector, while the map is trained by the explicit integrity BCE.
    """
    def __init__(self, bev_model, reliability):
        self.model = getattr(bev_model, 'module', bev_model)
        self.reliability = reliability
        if not hasattr(self.model, 'img_neck') or self.model.img_neck is None:
            raise ValueError("RAF currently requires an image FPN neck")
        self.teacher_state = copy.deepcopy(self.model.prev_frame_info)
        self.initial_state = copy.deepcopy(self.model.prev_frame_info)
        self.losses, self.maps = [], []

    @contextmanager
    def evaluation_state(self):
        """Validation starts a fresh stream and restores the student's training cache."""
        training_state = copy.deepcopy(self.model.prev_frame_info)
        self.model.prev_frame_info = copy.deepcopy(self.initial_state)
        try:
            yield
        finally:
            self.model.prev_frame_info = training_state

    @contextmanager
    def apply(self, augment=False):
        integrity = [None]
        self.losses, self.maps = [], []

        def before_backbone(module, inputs):
            if augment:
                images, integrity[0] = corrupt_normalized_images(inputs[0])
                return (images,)+inputs[1:]

        def after_neck(module, inputs, outputs):
            with torch.set_grad_enabled(augment):
                features, loss, maps = self.reliability(outputs, integrity[0])
            if augment:
                self.losses.append(loss)
            # Keep only scalars for logging; no large activation caches.
            self.maps.append(torch.stack([m.detach().mean() for m in maps]).mean())
            return features

        h1 = self.model.img_backbone.register_forward_pre_hook(before_backbone)
        h2 = self.model.img_neck.register_forward_hook(after_neck)
        try:
            yield self
        finally:
            h1.remove()
            h2.remove()

    def paired_bev(self, bev_model, batch):
        """Separate teacher/student temporal caches and copies of mutable can_bus metadata."""
        student_state = self.model.prev_frame_info
        teacher_batch, student_batch = dict(batch), dict(batch)
        teacher_batch['img_metas'] = copy.deepcopy(batch['img_metas'])
        student_batch['img_metas'] = copy.deepcopy(batch['img_metas'])
        with torch.no_grad():
            try:
                self.model.prev_frame_info = self.teacher_state
                target = bev_model(return_loss=False, only_bev=True, **teacher_batch).detach()
                self.teacher_state = copy.deepcopy(self.model.prev_frame_info)
            finally:
                self.model.prev_frame_info = student_state
            with self.apply(augment=True):
                source = bev_model(return_loss=False, only_bev=True, **student_batch).detach()
        if not self.losses:
            raise RuntimeError("RAF hooks were not executed; cannot supervise reliability")
        return source, target, torch.stack(self.losses).mean()


def metadata_path(directory):
    return os.path.join(checkpoint_root(directory), 'raf_config.json')


def checkpoint_root(path):
    """Accept a checkpoint root, its unet directory, or a weights file."""
    directory = os.path.dirname(path) if os.path.isfile(path) else path
    directory = os.path.normpath(directory)
    return os.path.dirname(directory) if os.path.basename(directory) == 'unet' else directory


def read_config(directory):
    path = metadata_path(directory)
    if not os.path.isfile(path):
        return None
    with open(path) as stream:
        config = json.load(stream)
    if config.get('version') != 1 or config.get('supervision') != 'synthetic_integrity_v1':
        raise ValueError("Unsupported RAF checkpoint configuration")
    return config


def check_checkpoint_mode(directory, use_raf, exact_resume=False):
    directory = checkpoint_root(directory)
    config = read_config(directory)
    if config is not None and not use_raf:
        raise ValueError("RAF checkpoint requires --use_raf; refusing ungated evaluation/resume")
    if use_raf and exact_resume and config is None:
        raise ValueError("Baseline checkpoint cannot exactly resume RAF. Use --pretrained_unet_checkpoint to fine-tune.")
    if config is not None and not os.path.isfile(os.path.join(directory, 'raf.pt')):
        raise FileNotFoundError("RAF metadata exists but raf.pt is missing")
    return config


def save_reliability(model, directory, loss_weight=1.0):
    os.makedirs(directory, exist_ok=True)
    config = dict(model.config, loss_weight=loss_weight)
    with open(metadata_path(directory), 'w') as stream:
        json.dump(config, stream, indent=2)
    torch.save(model.state_dict(), os.path.join(directory, 'raf.pt'))


def load_reliability(model, directory):
    directory = checkpoint_root(directory)
    config = read_config(directory)
    if config is None:
        raise ValueError("Missing RAF checkpoint metadata")
    for key, value in model.config.items():
        if config.get(key) != value:
            raise ValueError("RAF checkpoint configuration mismatch for "+key)
    # weights_only is unavailable on the user's PyTorch 1.10 runtime.
    model.load_state_dict(torch.load(os.path.join(directory, 'raf.pt'), map_location='cpu'), strict=True)


def strict_load_unet(model, directory, subfolder='unet'):
    from safetensors.torch import load_file
    from diffusers.utils.constants import SAFETENSORS_WEIGHTS_NAME
    path = directory if os.path.isfile(directory) else os.path.join(directory, subfolder, SAFETENSORS_WEIGHTS_NAME)
    if not os.path.isfile(path) and os.path.isdir(directory):
        path = os.path.join(directory, SAFETENSORS_WEIGHTS_NAME)
    model.load_state_dict(load_file(path, device='cpu'), strict=True)


def register_checkpoint_hooks(accelerator, use_raf=False, loss_weight=1.0):
    """Save each prepared model separately and keep optimizer/scheduler in Accelerate."""
    def save_hook(models, weights, directory):
        for model in models:
            raw = accelerator.unwrap_model(model)
            if accelerator.is_main_process:
                if isinstance(raw, CameraReliability):
                    save_reliability(raw, directory, loss_weight)
                else:
                    raw.save_pretrained(os.path.join(directory, 'unet'))
            weights.pop()

    def load_hook(models, directory):
        while models:
            raw = accelerator.unwrap_model(models.pop())
            if isinstance(raw, CameraReliability):
                load_reliability(raw, directory)
            elif use_raf:
                strict_load_unet(raw, directory)
            else:
                raw.from_pretrained(os.path.join(directory, 'unet'))

    accelerator.register_save_state_pre_hook(save_hook)
    accelerator.register_load_state_pre_hook(load_hook)
