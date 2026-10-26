"""CPU checks for the opt-in shape experiment; no nuScenes/compiled MMCV needed."""
import argparse
import copy
import importlib
import json
from pathlib import Path
import sys
import types

import numpy as np
import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'BEVFormer'))
from projects.bevdiffuser.shape_modules import (
    CLASS_GROUPS, ShapeSVM, ResidualBottleneck, install_bottlenecks)
from projects.bevdiffuser.shape_integration import (
    add_shape_arguments, restore_options, register_shape_metadata, configure_unet,
    validate, load_shape_unet, count_training_classes, configure_class_balance)

torch.set_num_threads(2)


def test_svm_initial_fusion_is_baseline():
    svm = ShapeSVM(32)
    baseline = torch.randn(2, 3, 10)
    fine, group = svm(torch.randn(2, 3, 32), torch.randn(2, 3, 10))
    assert torch.equal(svm.fuse(baseline, fine, group), baseline)
    assert CLASS_GROUPS[1] == CLASS_GROUPS[2] == CLASS_GROUPS[3]
    assert CLASS_GROUPS[6] == CLASS_GROUPS[7]
    assert CLASS_GROUPS[-1] == 6


def test_hinge_matches_known_margin():
    scores = torch.tensor([[3.0, 1.0, 4.0]])
    # target=0: competitor1 gives zero; competitor2 gives 2, average=1.
    assert ShapeSVM.hinge(scores, torch.tensor([0])).item() == 1.0


def test_svm_gradients_reach_features_and_predicted_dimensions():
    svm = ShapeSVM(32)
    nn.init.normal_(svm.fine.weight, std=0.01)
    nn.init.normal_(svm.group.weight, std=0.01)
    features = torch.randn(1, 5, 32, requires_grad=True)
    boxes = torch.randn(1, 5, 10, requires_grad=True)
    fine, group = svm(features, boxes)
    loss = svm.objective(fine, group, torch.tensor([0, 1, 7, 10, 10]),
                         torch.ones(5), torch.rand(1, 5))
    loss.backward()
    assert features.grad.abs().sum() > 0
    assert boxes.grad[..., [2, 3, 5]].abs().sum() > 0
    assert boxes.grad[..., [0, 1, 4, 6, 7, 8, 9]].abs().sum() == 0
    assert svm.fine.weight.grad.abs().sum() > 0


@pytest.mark.parametrize('all_background', [True, False])
def test_empty_or_ignored_scenes_are_finite(all_background):
    svm = ShapeSVM(8)
    fine, group = svm(torch.randn(2, 8), torch.randn(2, 10))
    weights = torch.ones(2) if all_background else torch.zeros(2)
    loss = svm.objective(fine, group, torch.full((2,), 10), weights, torch.rand(2))
    assert torch.isfinite(loss)
    loss.backward()
    assert svm.fine.weight.grad is not None


def test_optimizing_svm_reduces_objective():
    torch.manual_seed(12)
    svm = ShapeSVM(8)
    optimizer = torch.optim.SGD(svm.parameters(), lr=0.5)
    features, boxes = torch.randn(20, 8), torch.randn(20, 10)
    labels = torch.arange(20) % 10
    def objective():
        fine, group = svm(features, boxes)
        return svm.objective(fine, group, labels, torch.ones(20), torch.rand(20))
    before = objective().item()
    for _ in range(25):
        optimizer.zero_grad()
        objective().backward()
        optimizer.step()
    assert objective().item() < before


def test_rbf_random_fourier_features_approximate_gaussian_kernel():
    torch.manual_seed(4)
    gamma = 0.4
    svm = ShapeSVM(4, kernel='rbf', rbf_features=8192, rbf_gamma=gamma)
    x = torch.tensor([[0.0] * 10,
                      [0.2, -0.1, 0.3, 0.4, 0.1, -0.2, 0.0, 0.1, -0.4, 0.2]])
    mapped = svm.rbf_map(x)
    approximation = (mapped[0] * mapped[1]).sum()
    exact = torch.exp(-gamma * (x[0] - x[1]).square().sum())
    assert abs(approximation.item() - exact.item()) < 0.04


def test_rbf_svm_is_differentiable_and_linear_mode_keeps_legacy_shapes():
    svm = ShapeSVM(16, kernel='rbf', rbf_features=64)
    features = torch.randn(2, 3, 16, requires_grad=True)
    boxes = torch.randn(2, 3, 10, requires_grad=True)
    fine, group = svm(features, boxes)
    loss = svm.objective(fine, group, torch.tensor([0, 3, 10, 4, 8, 10]),
                         torch.ones(6), torch.rand(2, 3))
    loss.backward()
    assert features.grad.abs().sum() > 0
    assert boxes.grad[..., [2, 3, 5]].abs().sum() > 0
    assert svm.fine.weight.shape == (11, 64)
    assert 'rff_omega' in svm.state_dict()

    linear = ShapeSVM(16, kernel='linear')
    assert linear.fine.weight.shape == (11, 22)
    assert linear.group.weight.shape == (7, 22)
    assert 'rff_omega' not in linear.state_dict()


def test_bottleneck_identity_and_training():
    adapter = ResidualBottleneck(32, 4)
    inputs = torch.randn(2, 32, 5, 5, requires_grad=True)
    assert torch.equal(adapter(inputs), inputs)
    adapter(inputs).square().mean().backward()
    assert adapter.expand.weight.grad.abs().sum() > 0
    assert inputs.grad.abs().sum() > 0


def small_unet():
    from projects.bevdiffuser.layout_diffusion.layout_diffusion_unet import LayoutDiffusionUNetModel
    from projects.bevdiffuser.layout_diffusion.layout_encoder import LayoutTransformerEncoder
    layout = LayoutTransformerEncoder(layout_length=4, hidden_dim=32, output_dim=128,
        num_layers=1, num_heads=4, use_final_ln=True, num_classes_for_layout_object=12,
        mask_size_for_layout_object=0, used_condition_types=['obj_class', 'obj_bbox', 'is_valid_obj'],
        use_positional_embedding=False, resolution_to_attention=[], use_3d_bbox=True)
    # The original encoder constructs image-patch geometry with .cuda(); supply
    # equivalent geometry on CPU for this test, without editing the baseline.
    layout.resolution_to_attention = [4, 8, 16]
    for resolution in layout.resolution_to_attention:
        interval = 1.0 / resolution
        layout.image_patch_bbox_embedding['resolution{}'.format(resolution)] = torch.tensor([
            (interval*j + interval/2 - 0.5, -(interval*i + interval/2 - 0.5),
             0, interval, interval, 0, 0, 0, 0)
            for i in range(resolution) for j in range(resolution)])
    model = LayoutDiffusionUNetModel(layout_encoder=layout, in_channels=32,
        model_channels=32, out_channels=32, num_res_blocks=2, attention_ds=(4, 2, 1),
        channel_mult=(1, 2, 4), image_size=16, encoder_channels=32,
        num_heads=4, use_scale_shift_norm=True, resblock_updown=True,
        attention_block_type='ObjectAwareCrossAttention', use_positional_embedding_for_attention=True)
    # Baseline U-Net final layer is zero-initialized; make identity comparison nontrivial.
    nn.init.normal_(model.out[-1].weight, std=0.01)
    return model


@pytest.mark.parametrize('time_gate', [False, True])
def test_real_unet_identity_call_count_gradients_and_checkpoint(tmp_path, time_gate):
    from safetensors.torch import save_file
    torch.manual_seed(23)
    original = small_unet().eval()
    modified = copy.deepcopy(original)
    assert install_bottlenecks(modified, 4, time_gate=time_gate) == 19
    inputs = torch.randn(1, 32, 16, 16)
    condition = dict(obj_class=torch.ones(1, 4, dtype=torch.long),
                     obj_bbox=torch.zeros(1, 4, 9), is_valid_obj=torch.ones(1, 4))
    timestep = torch.tensor([7])
    with torch.no_grad():
        expected = original(inputs, timestep, **condition)[0]
        actual = modified(inputs, timestep, **condition)[0]
    assert expected.abs().sum() > 0
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    calls = {key: 0 for key in modified.shape_adapters}
    hooks = []
    for key, adapter in modified.shape_adapters.items():
        def count(module, args, result, name=key):
            calls[name] += 1
        hooks.append(adapter.register_forward_hook(count))
    # Exercise two denoising calls at different timesteps, as inference would.
    for t in (3, 9):
        modified(inputs, torch.tensor([t]), **condition)[0].square().mean().backward()
    assert set(calls.values()) == {2}
    assert all(adapter.expand.weight.grad is not None for adapter in modified.shape_adapters.values())
    assert sum(adapter.expand.weight.grad.abs().sum().item()
               for adapter in modified.shape_adapters.values()) > 0
    for hook in hooks:
        hook.remove()
    save_file(modified.state_dict(), str(tmp_path / 'diffusion_pytorch_model.safetensors'))
    restored = small_unet().eval()
    install_bottlenecks(restored, 4, time_gate=time_gate)
    load_shape_unet(restored, str(tmp_path))
    torch.testing.assert_close(restored(inputs, timestep, **condition)[0],
                               modified(inputs, timestep, **condition)[0])
    bare = small_unet()
    with pytest.raises(ValueError, match='mismatch'):
        load_shape_unet(bare, str(tmp_path))


def test_baseline_unet_initialization_allows_only_new_adapters(tmp_path):
    from safetensors.torch import save_file
    model = small_unet()
    state = model.state_dict()
    path = tmp_path / 'diffusion_pytorch_model.safetensors'
    save_file(state, str(path))
    install_bottlenecks(model)
    load_shape_unet(model, str(tmp_path), allow_baseline=True)
    with pytest.raises(ValueError, match='mismatch'):
        load_shape_unet(model, str(tmp_path))
    state.pop('time_embed.0.weight')
    save_file(state, str(path))
    with pytest.raises(ValueError, match='time_embed'):
        load_shape_unet(model, str(tmp_path), allow_baseline=True)


def load_augmentation(monkeypatch):
    # Only the decorator is mocked; image conversion and insertion are the real code.
    registry = types.SimpleNamespace(register_module=lambda: lambda cls: cls)
    for name in ('mmdet', 'mmdet.datasets', 'mmdet.datasets.builder'):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    sys.modules['mmdet.datasets.builder'].PIPELINES = registry
    name = 'projects.bevdiffuser.shape_augmentation'
    monkeypatch.delitem(sys.modules, name, raising=False)
    return importlib.import_module(name)


def test_grayscale_bgr_channels_dtype_geometry_and_order(monkeypatch):
    aug = load_augmentation(monkeypatch)
    images = [np.full((3, 4, 3), [0, 0, 100 + i], dtype=np.float32) for i in range(6)]
    result = dict(img=images, lidar2img=['calibration'] * 6)
    output = aug.RandomGrayscaleMultiView(1)(result)
    for i, image in enumerate(output['img']):
        assert image.shape == (3, 4, 3) and image.dtype == np.float32
        np.testing.assert_allclose(image, 0.299 * (100 + i))
        np.testing.assert_array_equal(image[..., 0], image[..., 2])
    assert output['lidar2img'] == ['calibration'] * 6
    unchanged = dict(img=images)
    assert aug.RandomGrayscaleMultiView(0)(unchanged)['img'] is images


def test_grayscale_insertion_does_not_touch_validation(monkeypatch):
    aug = load_augmentation(monkeypatch)
    pipeline = [dict(type='LoadMultiViewImageFromFiles'), dict(type='PhotoMetricDistortionMultiViewImage'),
                dict(type='NormalizeMultiviewImage'), dict(type='PadMultiViewImage')]
    train, test = dict(pipeline=copy.deepcopy(pipeline)), dict(pipeline=copy.deepcopy(pipeline))
    aug.enable_grayscale(train, 0.3)
    assert train['pipeline'][2]['type'] == 'RandomGrayscaleMultiView'
    assert test['pipeline'] == pipeline


def test_checkpoint_restores_architecture_and_rejects_missing_modules(tmp_path):
    parser = argparse.ArgumentParser()
    add_shape_arguments(parser)
    args = parser.parse_args(['--use_svm', '--grayscale_aug', '--use_unet_bottleneck'])
    hooks = []
    accelerator = types.SimpleNamespace(is_main_process=True,
        register_save_state_pre_hook=lambda hook: hooks.append(hook))
    register_shape_metadata(accelerator, args)
    hooks[0]([], [], str(tmp_path))
    restored = parser.parse_args([])
    restore_options(restored, str(tmp_path), exact=True)
    assert restored.use_svm and restored.use_unet_bottleneck and restored.grayscale_aug
    bare = tmp_path / 'baseline'
    bare.mkdir()
    with pytest.raises(ValueError, match='No shape_options'):
        restore_options(restored, str(bare), exact=True)


def test_v2_checkpoint_migrates_to_legacy_linear_svm(tmp_path):
    parser = argparse.ArgumentParser()
    add_shape_arguments(parser)
    args = parser.parse_args(['--use_svm'])
    hooks = []
    accelerator = types.SimpleNamespace(is_main_process=True,
        register_save_state_pre_hook=lambda hook: hooks.append(hook))
    register_shape_metadata(accelerator, args)
    hooks[0]([], [], str(tmp_path))
    path = tmp_path / 'shape_options.json'
    saved = json.loads(path.read_text())
    saved['version'] = 2
    for key in ('svm_kernel', 'svm_rbf_features', 'svm_rbf_gamma'):
        saved['options'].pop(key)
    path.write_text(json.dumps(saved))
    restored = parser.parse_args(['--use_svm'])
    restore_options(restored, str(tmp_path), exact=True)
    assert restored.svm_kernel == 'linear'


def test_disabled_flags_leave_unet_untouched():
    parser = argparse.ArgumentParser()
    add_shape_arguments(parser)
    args = parser.parse_args([])
    model = nn.Linear(3, 4)
    before = set(model.state_dict())
    assert configure_unet(model, args) == 0
    assert set(model.state_dict()) == before


def test_svm_requires_detection_gradient_path():
    parser = argparse.ArgumentParser()
    add_shape_arguments(parser)
    args = parser.parse_args(['--use_svm'])
    args.task_loss_scale, args.prediction_type = 0, 'sample'
    with pytest.raises(ValueError, match='task_loss_scale'):
        validate(args, training=True)


def test_time_gate_preserves_identity_then_varies_with_embedding_and_trains():
    adapter = ResidualBottleneck(32, 4, time_emb_dims=128)
    inputs = torch.randn(2, 32, 4, 4)
    embedding = torch.randn(2, 128, requires_grad=True)
    assert torch.equal(adapter(inputs, embedding), inputs)
    with torch.no_grad():
        adapter.expand.bias.fill_(1)
        adapter.time_gate[-1].weight.fill_(0.01)
    low = torch.zeros(2, 128)
    high = torch.ones(2, 128)
    assert not torch.equal(adapter(inputs, low), adapter(inputs, high))
    adapter(inputs, embedding).square().mean().backward()
    assert embedding.grad.abs().sum() > 0
    assert adapter.time_gate[-1].weight.grad.abs().sum() > 0
    with pytest.raises(ValueError, match='embedding'):
        adapter(inputs)


def test_class_balancing_upweights_rare_classes_and_preserves_background(tmp_path):
    svm = ShapeSVM(8, class_balance=True)
    counts = [10000, 100, 10, 100, 100, 2000, 100, 100, 5000, 0]
    svm.set_class_counts(counts)
    assert svm.fine_class_weights[2] > svm.fine_class_weights[0]
    assert svm.fine_class_weights[9] == 0
    assert svm.fine_class_weights[10] == 1 and svm.group_class_weights[6] == 1
    assert svm.fine_class_weights.max() <= 4
    fine = torch.zeros(2, 11, requires_grad=True)
    group = torch.zeros(2, 7, requires_grad=True)
    loss = svm.objective(fine, group, torch.tensor([0, 2]), torch.ones(2), torch.zeros(2))
    loss.backward()
    assert fine.grad[1, 2].abs() > fine.grad[0, 0].abs()
    restored = ShapeSVM(8, class_balance=True)
    restored.load_state_dict(svm.state_dict(), strict=True)
    assert torch.equal(restored.class_counts, svm.class_counts)
    assert torch.equal(restored.fine_class_weights, svm.fine_class_weights)
    unbalanced = ShapeSVM(8)
    unbalanced.set_class_counts(counts)
    assert torch.equal(unbalanced.fine_class_weights, torch.ones(11))


@pytest.mark.parametrize('label_dtype', [np.int64, np.float64])
def test_training_counts_use_annotations_and_range_only(label_dtype):
    class Boxes:
        def in_range_bev(self, bounds):
            assert bounds == [-50, -50, 50, 50]
            return torch.tensor(self.mask, dtype=torch.bool)
        def __init__(self, mask):
            self.mask = mask
    class Dataset:
        CLASSES = ('car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                   'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone')
        def __len__(self):
            return 3
        def get_ann_info(self, index):
            if index == 0:
                return dict(gt_labels_3d=np.array([]), gt_bboxes_3d=Boxes([]))
            return dict(gt_labels_3d=np.array([0, 2, 1, -1], dtype=label_dtype),
                        gt_bboxes_3d=Boxes([True, True, False, True]))
        def __getitem__(self, index):
            raise AssertionError('Counting must not load images or the dataset pipeline')
    dataset = Dataset()
    counts = count_training_classes(dataset, [-50, -50, -5, 50, 50, 3])
    np.testing.assert_array_equal(counts, [2, 0, 2, 0, 0, 0, 0, 0, 0, 0])
    parser = argparse.ArgumentParser()
    add_shape_arguments(parser)
    args = parser.parse_args(['--use_svm', '--svm_class_balance'])
    svm = ShapeSVM(8, class_balance=True)
    model = types.SimpleNamespace(pts_bbox_head=types.SimpleNamespace(shape_svm=svm))
    configure_class_balance(model, dataset, args, [-50, -50, -5, 50, 50, 3])
    args._shape_exact = True
    # A resumed checkpoint uses saved counts, with no training-data recount.
    configure_class_balance(model, None, args, None)
    assert args._svm_class_counts == counts.tolist()


@pytest.mark.parametrize('flag', ['--svm_class_balance', '--bottleneck_time_gate'])
def test_extra_flags_require_their_parent_module(flag):
    parser = argparse.ArgumentParser()
    add_shape_arguments(parser)
    args = parser.parse_args([flag])
    with pytest.raises(ValueError, match='requires'):
        validate(args)
