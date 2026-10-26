import types
import unittest

import torch
from torch import nn

from temporal_cross_attention import (
    ConfidenceGatedTemporalCrossAttention,
    configure_temporal_cross_attention,
)


class _Middle(nn.Module):
    channels = 16

    def forward(self, value, *_args, **_kwargs):
        return value, None


class _Transformer:
    def get_bev_features(self, prev_bev=None, **_kwargs):
        return torch.zeros(1)


class _Detector(nn.Module):
    def __init__(self):
        super().__init__()
        self.pts_bbox_head = types.SimpleNamespace(transformer=_Transformer())


class _Wrapper:
    def __init__(self, module):
        self.module = module


class _UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.middle_block = _Middle()


class TemporalCrossAttentionTests(unittest.TestCase):
    def test_zero_initialized_residual_is_identity(self):
        module = ConfidenceGatedTemporalCrossAttention(
            channels=16, bev_channels=8, heads=4
        )
        current = torch.randn(2, 16, 3, 3)
        previous = torch.randn(2, 16, 8)
        actual = module(current, previous, bev_height=4, bev_width=4)
        torch.testing.assert_close(actual, current)

    def test_attention_residual_and_confidence_are_differentiable(self):
        module = ConfidenceGatedTemporalCrossAttention(
            channels=16, bev_channels=8, heads=4
        )
        with torch.no_grad():
            module.output_proj.weight.normal_(mean=0.0, std=0.02)
        current = torch.randn(2, 16, 3, 3, requires_grad=True)
        previous = torch.randn(2, 16, 8)
        output = module(current, previous, bev_height=4, bev_width=4)
        self.assertEqual(tuple(output.shape), tuple(current.shape))
        output.square().mean().backward()
        self.assertIsNotNone(module.confidence[-1].weight.grad)
        self.assertIsNotNone(module.key_proj.weight.grad)

    def test_runtime_captures_and_uses_previous_bev(self):
        unet = _UNet()
        detector = _Wrapper(_Detector())
        cfg = types.SimpleNamespace(bev_h_=4, bev_w_=4, _dim_=8)
        args = types.SimpleNamespace(
            use_temporal_cross_attention=True,
            temporal_attention_heads=4,
            temporal_attention_dropout=0.0,
        )
        runtime = configure_temporal_cross_attention(unet, detector, cfg, args)
        runtime.begin_bev_call()
        previous = torch.randn(16, 2, 8)
        detector.module.pts_bbox_head.transformer.get_bev_features(prev_bev=previous)
        current = torch.randn(2, 8, 4, 4)
        self.assertTrue(runtime.use_captured_context(current))
        self.assertEqual(tuple(runtime.context.shape), (2, 16, 8))

        before = torch.randn(2, 16, 2, 2)
        after, _ = unet.middle_block(before, None)
        torch.testing.assert_close(after, before)

    def test_new_scene_without_history_bypasses_attention(self):
        unet = _UNet()
        detector = _Wrapper(_Detector())
        cfg = types.SimpleNamespace(bev_h_=4, bev_w_=4, _dim_=8)
        args = types.SimpleNamespace(
            use_temporal_cross_attention=True,
            temporal_attention_heads=4,
            temporal_attention_dropout=0.0,
        )
        runtime = configure_temporal_cross_attention(unet, detector, cfg, args)
        runtime.begin_bev_call()
        self.assertFalse(runtime.use_captured_context(torch.randn(1, 8, 4, 4)))
        value = torch.randn(1, 16, 2, 2)
        output, _ = unet.middle_block(value, None)
        torch.testing.assert_close(output, value)


if __name__ == "__main__":
    unittest.main()
