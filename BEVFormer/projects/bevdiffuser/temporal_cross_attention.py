# SPDX-License-Identifier: AGPL-3.0
"""Optional confidence-gated temporal cross-attention for BEVDiffuser.

The module consumes the previous BEV feature that BEVFormer already uses for
temporal self-attention, and injects it once at the diffusion U-Net middle
block. A learned per-location confidence gate scales the attention residual.
"""
import inspect
import json
import os

import torch
from torch import nn


CHECKPOINT_FILE = "temporal_cross_attention.json"
MODULE_NAME = "temporal_cross_attention"
DEFAULT_HEADS = 4
DEFAULT_DROPOUT = 0.0


def add_temporal_arguments(parser):
    parser.add_argument(
        "--use_temporal_cross_attention", action="store_true",
        help="Use confidence-gated previous-frame BEV attention at the U-Net bottleneck",
    )
    parser.add_argument(
        "--temporal_attention_heads", type=int, default=DEFAULT_HEADS,
        help="Number of temporal cross-attention heads (default: 4)",
    )
    parser.add_argument(
        "--temporal_attention_dropout", type=float, default=DEFAULT_DROPOUT,
        help="Attention dropout for temporal cross-attention (default: 0)",
    )


def _metadata_path(directory):
    if not directory or not os.path.isdir(directory):
        return None
    candidate = os.path.join(directory, CHECKPOINT_FILE)
    return candidate if os.path.isfile(candidate) else None


def restore_temporal_options(args, directory, exact=False):
    """Restore architecture options so resume and evaluation match the checkpoint."""
    path = _metadata_path(directory)
    if path is None:
        if exact and args.use_temporal_cross_attention:
            raise ValueError(
                "The requested temporal-attention checkpoint has no " + CHECKPOINT_FILE
            )
        return
    with open(path, "r") as handle:
        saved = json.load(handle)
    if saved.get("version") != 1:
        raise ValueError("Unsupported temporal-attention checkpoint metadata version")
    enabled = saved.get("enabled")
    if not isinstance(enabled, bool):
        raise ValueError("Invalid temporal-attention checkpoint metadata")
    if exact and args.use_temporal_cross_attention and not enabled:
        raise ValueError("This checkpoint was not trained with temporal cross-attention")
    args.use_temporal_cross_attention = enabled
    if enabled:
        args.temporal_attention_heads = int(saved["heads"])
        args.temporal_attention_dropout = float(saved["dropout"])
    validate_temporal_options(args)


def validate_temporal_options(args):
    if args.temporal_attention_heads < 1:
        raise ValueError("--temporal_attention_heads must be a positive integer")
    if not 0.0 <= args.temporal_attention_dropout < 1.0:
        raise ValueError("--temporal_attention_dropout must be in [0, 1)")


class ConfidenceGatedTemporalCrossAttention(nn.Module):
    """Attend from current U-Net bottleneck locations to previous BEV tokens."""

    def __init__(self, channels, bev_channels, heads=DEFAULT_HEADS, dropout=DEFAULT_DROPOUT):
        super().__init__()
        if channels < 1 or bev_channels < 1 or heads < 1:
            raise ValueError("Channel counts and attention heads must be positive")
        self.channels = int(channels)
        self.bev_channels = int(bev_channels)
        self.heads = int(heads)
        self.attention_dim = max(self.heads, min(self.channels, 128))
        self.attention_dim = ((self.attention_dim + self.heads - 1) // self.heads) * self.heads

        self.query_norm = nn.LayerNorm(self.channels)
        self.memory_norm = nn.LayerNorm(self.bev_channels)
        self.query_proj = nn.Linear(self.channels, self.attention_dim)
        self.key_proj = nn.Linear(self.bev_channels, self.attention_dim)
        self.value_proj = nn.Linear(self.bev_channels, self.attention_dim)
        self.position_proj = nn.Linear(2, self.attention_dim, bias=False)
        self.attention = nn.MultiheadAttention(
            self.attention_dim, self.heads, dropout=float(dropout), batch_first=True
        )
        self.confidence = nn.Sequential(
            nn.Linear(self.attention_dim * 3, self.attention_dim),
            nn.SiLU(),
            nn.Linear(self.attention_dim, 1),
        )
        self.output_proj = nn.Linear(self.attention_dim, self.channels)

        # Start as an exact no-op; optimization learns the attention residual.
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    @staticmethod
    def _coordinates(height, width, device, dtype):
        y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
        x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
        yy = y[:, None].expand(height, width)
        xx = x[None, :].expand(height, width)
        return torch.stack((xx, yy), dim=-1).reshape(1, height * width, 2)

    def forward(self, features, previous_bev, bev_height, bev_width):
        if features.ndim != 4:
            raise ValueError("U-Net bottleneck features must have shape [B,C,H,W]")
        if previous_bev.ndim != 3:
            raise ValueError("Previous BEV context must have shape [B,H*W,C]")
        batch, channels, height, width = features.shape
        if channels != self.channels:
            raise ValueError("Unexpected U-Net bottleneck channel count")
        if previous_bev.shape[0] != batch:
            raise ValueError(
                "Previous BEV batch size {} does not match U-Net batch size {}".format(
                    previous_bev.shape[0], batch
                )
            )
        if previous_bev.shape[1] != bev_height * bev_width:
            raise ValueError("Previous BEV token count does not match configured BEV size")
        if previous_bev.shape[2] != self.bev_channels:
            raise ValueError("Previous BEV feature width does not match the model config")

        current = features.flatten(2).transpose(1, 2)
        memory = previous_bev
        query = self.query_proj(self.query_norm(current))
        key_value = self.memory_norm(memory)
        key = self.key_proj(key_value)
        value = self.value_proj(key_value)

        query_pos = self._coordinates(height, width, features.device, query.dtype)
        memory_pos = self._coordinates(bev_height, bev_width, features.device, query.dtype)
        query = query + self.position_proj(query_pos)
        key = key + self.position_proj(memory_pos)

        attended, _ = self.attention(query, key, value, need_weights=False)
        confidence_input = torch.cat((query, attended, torch.abs(query - attended)), dim=-1)
        gate = torch.sigmoid(self.confidence(confidence_input))
        residual = self.output_proj(attended) * gate
        residual = residual.transpose(1, 2).reshape(batch, channels, height, width)
        return features + residual.to(features.dtype)


class _TemporalRuntime:
    """Shares the captured BEV history between BEVFormer and U-Net hooks."""

    def __init__(self, unet, bev_model, attention_module, bev_height, bev_width, bev_channels):
        self.unet = unet
        self.bev_model = bev_model
        self.attention_module = attention_module
        self.bev_height = int(bev_height)
        self.bev_width = int(bev_width)
        self.bev_channels = int(bev_channels)
        self._captured = None
        self.context = None
        self._transformer = None
        self._original_get_bev_features = None
        self._attach_capture()
        self._attach_unet_hook()

    @staticmethod
    def _unwrap(model):
        seen = set()
        while hasattr(model, "module") and id(model) not in seen:
            seen.add(id(model))
            model = model.module
        return model

    def _attach_capture(self):
        model = self._unwrap(self.bev_model)
        if model is None:
            raise RuntimeError("Temporal runtime was not given the BEVFormer model")
        head = getattr(model, "pts_bbox_head", None)
        transformer = getattr(head, "transformer", None) if head is not None else None
        if transformer is None or not callable(getattr(transformer, "get_bev_features", None)):
            raise RuntimeError(
                "Could not locate BEVFormer pts_bbox_head.transformer.get_bev_features"
            )
        original = transformer.get_bev_features
        signature = inspect.signature(original)

        def capture_previous(*args, **kwargs):
            bound = signature.bind_partial(*args, **kwargs)
            previous = bound.arguments.get("prev_bev")
            if previous is None:
                extra_kwargs = bound.arguments.get("kwargs")
                if isinstance(extra_kwargs, dict):
                    previous = extra_kwargs.get("prev_bev")
            if previous is None:
                previous = kwargs.get("prev_bev")
            result = original(*args, **kwargs)
            if previous is not None:
                if not torch.is_tensor(previous):
                    raise TypeError("BEVFormer prev_bev must be a tensor")
                self._captured = previous.detach().clone()
            return result

        transformer.get_bev_features = capture_previous
        self._transformer = transformer
        self._original_get_bev_features = original

    def _attach_unet_hook(self):
        block = getattr(self.unet, "middle_block", None)
        if block is None:
            raise RuntimeError("Could not locate U-Net middle_block")

        def apply_temporal(_module, _inputs, output):
            if self.context is None:
                return output
            if not isinstance(output, tuple) or len(output) < 1:
                raise RuntimeError("U-Net middle_block must return (features, extra_output)")
            features = output[0]
            enhanced = self.attention_module(
                features, self.context, self.bev_height, self.bev_width
            )
            return (enhanced,) + output[1:]

        block.register_forward_hook(apply_temporal)

    def begin_bev_call(self):
        self._captured = None
        self.context = None

    def use_captured_context(self, current_latents):
        if self._captured is None:
            self.context = None
            return False
        previous = self._captured
        tokens = self.bev_height * self.bev_width
        if previous.ndim != 3:
            raise ValueError("Captured BEV history must have rank 3")
        if previous.shape[0] == tokens:
            previous = previous.permute(1, 0, 2)
        elif previous.shape[1] != tokens:
            raise ValueError(
                "Cannot identify BEV token dimension in captured shape {}".format(
                    tuple(previous.shape)
                )
            )
        if previous.shape[0] != current_latents.shape[0]:
            raise ValueError(
                "Captured temporal batch {} does not match current batch {}".format(
                    previous.shape[0], current_latents.shape[0]
                )
            )
        if previous.shape[2] != self.bev_channels:
            raise ValueError("Captured BEV channels do not match the configured BEV width")
        self.context = previous.to(device=current_latents.device, dtype=current_latents.dtype)
        return True

    def clear_context(self):
        self.context = None
        self._captured = None


def _model_channels(block):
    layers = list(block.children())
    if layers:
        layer = layers[-1]
        channels = getattr(layer, "out_channels", getattr(layer, "channels", None))
        if channels is not None:
            return int(channels)
    channels = getattr(block, "out_channels", getattr(block, "channels", None))
    return int(channels) if channels is not None else None


def _find_temporal_transformer(bev_model):
    model = _TemporalRuntime._unwrap(bev_model)
    head = getattr(model, "pts_bbox_head", None)
    transformer = getattr(head, "transformer", None) if head is not None else None
    return transformer


def configure_temporal_cross_attention(unet, bev_model, bev_cfg, args):
    validate_temporal_options(args)
    if not args.use_temporal_cross_attention:
        return None
    if hasattr(unet, MODULE_NAME):
        raise ValueError("Temporal cross-attention is already configured on this U-Net")
    transformer = _find_temporal_transformer(bev_model)
    if transformer is None or not callable(getattr(transformer, "get_bev_features", None)):
        raise RuntimeError("BEVFormer transformer does not expose get_bev_features")
    middle = getattr(unet, "middle_block", None)
    channels = _model_channels(middle) if middle is not None else None
    if channels is None:
        raise RuntimeError("Cannot infer channel width for the U-Net middle_block")
    module = ConfidenceGatedTemporalCrossAttention(
        channels=channels,
        bev_channels=int(bev_cfg._dim_),
        heads=args.temporal_attention_heads,
        dropout=args.temporal_attention_dropout,
    )
    unet.add_module(MODULE_NAME, module)
    runtime = _TemporalRuntime(
        unet, bev_model, module, bev_cfg.bev_h_, bev_cfg.bev_w_, bev_cfg._dim_
    )
    unet._temporal_attention_runtime = runtime
    return runtime


def get_temporal_runtime(unet):
    while hasattr(unet, "module"):
        unet = unet.module
    return getattr(unet, "_temporal_attention_runtime", None)


def set_temporal_trainable(unet, args):
    if args.use_temporal_cross_attention:
        module = getattr(_TemporalRuntime._unwrap(unet), MODULE_NAME, None)
        if module is None:
            raise RuntimeError("Temporal cross-attention was enabled but not configured")
        module.requires_grad_(True)


def _checkpoint_state_path(directory):
    candidates = [
        os.path.join(directory, "diffusion_pytorch_model.safetensors"),
        os.path.join(directory, "unet", "diffusion_pytorch_model.safetensors"),
    ]
    return next((path for path in candidates if os.path.isfile(path)), None)


def load_temporal_weights(unet, directory, required=False):
    """Load and validate only the optional temporal module's saved parameters."""
    unet = _TemporalRuntime._unwrap(unet)
    module = getattr(unet, MODULE_NAME, None)
    path = _metadata_path(directory)
    if module is None:
        if required:
            raise ValueError("Temporal checkpoint requested but its U-Net module is not configured")
        return False
    if path is None:
        if required:
            raise ValueError("Missing temporal checkpoint metadata in {}".format(directory))
        return False
    with open(path, "r") as handle:
        metadata = json.load(handle)
    if metadata.get("version") != 1 or not metadata.get("enabled"):
        if required:
            raise ValueError("Checkpoint metadata does not enable temporal cross-attention")
        return False
    state_path = _checkpoint_state_path(directory)
    if state_path is None:
        raise FileNotFoundError("Cannot find U-Net safetensors in {}".format(directory))
    from safetensors.torch import load_file
    state = load_file(state_path, device="cpu")
    prefix = MODULE_NAME + "."
    temporal_state = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
    expected = module.state_dict()
    missing = set(expected) - set(temporal_state)
    unexpected = set(temporal_state) - set(expected)
    if missing or unexpected:
        raise ValueError(
            "Temporal checkpoint parameter mismatch; missing {}, unexpected {}".format(
                sorted(missing), sorted(unexpected)
            )
        )
    module.load_state_dict(temporal_state, strict=True)
    return True


def load_pretrained_with_temporal(unet, directory, loader, required=False):
    """Temporarily detach the add-on while loading the original U-Net weights."""
    base_unet = _TemporalRuntime._unwrap(unet)
    module = base_unet._modules.get(MODULE_NAME)
    metadata_path = _metadata_path(directory)
    checkpoint_has_temporal = False
    if metadata_path:
        with open(metadata_path, "r") as handle:
            checkpoint_has_temporal = bool(json.load(handle).get("enabled"))
    if required and module is not None and not checkpoint_has_temporal:
        raise ValueError("Resume checkpoint was not saved with temporal cross-attention")

    # Shape-aware loading checks exact state-dict keys. Keep the temporal child
    # registered only when its weights are present; otherwise load the baseline
    # U-Net first and retain the freshly initialized residual adapter.
    if module is not None and not checkpoint_has_temporal:
        base_unet._modules.pop(MODULE_NAME)
        try:
            result = loader()
        finally:
            base_unet.add_module(MODULE_NAME, module)
    else:
        result = loader()
    if module is not None and checkpoint_has_temporal:
        load_temporal_weights(base_unet, directory, required=True)
    return result


def register_temporal_metadata(accelerator, args):
    if not args.use_temporal_cross_attention:
        return
    saved = {
        "version": 1,
        "enabled": True,
        "heads": int(args.temporal_attention_heads),
        "dropout": float(args.temporal_attention_dropout),
        "injection": "U-Net middle_block",
        "context": "BEVFormer previous-frame prev_bev",
    }

    def save_hook(_models, _weights, output_dir):
        if accelerator.is_main_process:
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, CHECKPOINT_FILE), "w") as handle:
                json.dump(saved, handle, indent=2)

    accelerator.register_save_state_pre_hook(save_hook)
