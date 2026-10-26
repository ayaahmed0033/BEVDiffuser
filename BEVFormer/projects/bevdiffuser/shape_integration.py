# SPDX-License-Identifier: AGPL-3.0
"""Flags, checkpoint identity and opt-in wiring for the shape experiment."""
import json
import os

from .shape_modules import CLASS_NAMES, install_bottlenecks


OPTION_DEFAULTS = dict(use_svm=False, grayscale_aug=False, grayscale_prob=0.3,
                       use_unet_bottleneck=False, bottleneck_ratio=4,
                       bottleneck_time_gate=False, svm_class_balance=False,
                       svm_fusion_weight=0.1, svm_loss_weight=0.1, svm_l2_weight=1e-4,
                       svm_kernel='rbf', svm_rbf_features=512, svm_rbf_gamma=None)


def add_shape_arguments(parser):
    parser.add_argument('--use_svm', action='store_true', help='Joint fine/coarse SVM heads')
    parser.add_argument('--grayscale_aug', action='store_true', help='Random camera grayscale during training')
    parser.add_argument('--grayscale_prob', type=float, default=0.3)
    parser.add_argument('--use_unet_bottleneck', action='store_true', help='Residual adapters after U-Net blocks')
    parser.add_argument('--bottleneck_ratio', type=int, default=4)
    parser.add_argument('--bottleneck_time_gate', action='store_true', help='Condition adapter strength on timestep/layout embedding')
    parser.add_argument('--svm_class_balance', action='store_true', help='Weight SVM positive losses using training class frequency')
    parser.add_argument('--svm_fusion_weight', type=float, default=0.1)
    parser.add_argument('--svm_loss_weight', type=float, default=0.1)
    parser.add_argument('--svm_l2_weight', type=float, default=1e-4)
    parser.add_argument('--svm_kernel', choices=('rbf', 'linear'), default='rbf',
                        help='RBF uses differentiable random Fourier features')
    parser.add_argument('--svm_rbf_features', type=int, default=512,
                        help='Number of fixed Fourier features used for the RBF approximation')
    parser.add_argument('--svm_rbf_gamma', type=float, default=None,
                        help='RBF gamma; default is 1/(descriptor dimension)')


def active(args):
    return args.use_svm or args.grayscale_aug or args.use_unet_bottleneck


def options(args):
    return {key: getattr(args, key) for key in OPTION_DEFAULTS}


def validate(args, training=False):
    if args.bottleneck_time_gate and not args.use_unet_bottleneck:
        raise ValueError('--bottleneck_time_gate requires --use_unet_bottleneck')
    if args.svm_class_balance and not args.use_svm:
        raise ValueError('--svm_class_balance requires --use_svm')
    if not 0 <= args.grayscale_prob <= 1 or args.bottleneck_ratio < 1:
        raise ValueError('Invalid grayscale probability or bottleneck ratio')
    if args.svm_rbf_features < 1 or (args.svm_rbf_gamma is not None and args.svm_rbf_gamma <= 0):
        raise ValueError('RBF feature count and gamma must be positive')
    if min(args.svm_fusion_weight, args.svm_l2_weight) < 0 or args.svm_loss_weight <= 0:
        raise ValueError('SVM loss must be positive; fusion and L2 weights must be nonnegative')
    if training and args.use_svm and (args.task_loss_scale <= 0 or args.prediction_type != 'sample'):
        raise ValueError('--use_svm requires positive --task_loss_scale and --prediction_type sample')


def restore_options(args, directory, exact=False):
    if not directory or not os.path.isdir(directory):
        if exact and active(args):
            raise ValueError('Shape experiment requires a checkpoint directory')
        return
    path = os.path.join(directory, 'shape_options.json')
    if os.path.isfile(path):
        with open(path) as handle:
            saved = json.load(handle)
        version = saved.get('version')
        if version not in (2, 3) or saved.get('class_names') != list(CLASS_NAMES):
            raise ValueError('Unsupported shape checkpoint version or class order')
        stored = dict(saved['options'])
        if version == 2:
            # v2 checkpoints used the original primal linear SVM and need the
            # matching head width when their saved model weights are restored.
            stored.update(svm_kernel='linear', svm_rbf_features=512, svm_rbf_gamma=None)
        for key in OPTION_DEFAULTS:
            if key not in stored:
                raise ValueError('Incomplete shape checkpoint: {}'.format(key))
            # Explicitly asking to add a missing architecture during exact load is an error.
            if exact and key in ('use_svm', 'use_unet_bottleneck', 'bottleneck_time_gate', 'svm_class_balance'):
                if getattr(args, key) and not stored[key]:
                    raise ValueError('Checkpoint does not contain {}'.format(key))
            setattr(args, key, stored[key])
    elif exact and active(args):
        raise ValueError('No shape_options.json: this checkpoint did not train the requested modules')


def configure_model(cfg, args):
    if not getattr(args, 'use_svm', False):
        return
    if tuple(cfg.class_names) != CLASS_NAMES or cfg.model.pts_bbox_head.type != 'BEVFormerHead':
        raise ValueError('SVM integration currently supports the standard nuScenes BEVFormerHead')
    from projects.mmdet3d_plugin.bevformer.dense_heads import shape_svm_head
    cfg.model.pts_bbox_head.type = 'ShapeSVMHead'
    cfg.model.pts_bbox_head.svm_options = dict(
        fusion_weight=args.svm_fusion_weight, loss_weight=args.svm_loss_weight,
        l2_weight=args.svm_l2_weight, class_balance=args.svm_class_balance,
        kernel=args.svm_kernel, rbf_features=args.svm_rbf_features,
        rbf_gamma=args.svm_rbf_gamma)


def configure_training_data(cfg, args):
    if args.grayscale_aug:
        from .shape_augmentation import enable_grayscale
        enable_grayscale(cfg.data.train, args.grayscale_prob)


def configure_unet(unet, args):
    if args.use_unet_bottleneck:
        return install_bottlenecks(unet, args.bottleneck_ratio, time_gate=args.bottleneck_time_gate)
    return 0


def load_shape_unet(unet, directory, allow_baseline=False):
    from safetensors.torch import load_file
    filename = 'diffusion_pytorch_model.safetensors'
    candidates = [directory, os.path.join(directory, filename),
                  os.path.join(directory, 'unet', filename)]
    path = next((p for p in candidates if os.path.isfile(p)), None)
    if path is None:
        raise FileNotFoundError('Cannot find U-Net safetensors in {}'.format(directory))
    state = load_file(path, device='cpu')
    expected = unet.state_dict()
    missing = set(expected) - set(state)
    unexpected = set(state) - set(expected)
    allowed = {k for k in missing if k.startswith('shape_adapters.')} if allow_baseline else set()
    if missing != allowed or unexpected:
        raise ValueError('U-Net checkpoint mismatch: missing {}, unexpected {}'.format(
            sorted(missing - allowed), sorted(unexpected)))
    unet.load_state_dict(state, strict=not bool(allowed))


def register_shape_metadata(accelerator, args):
    if not active(args):
        return
    saved = dict(version=3, class_names=list(CLASS_NAMES), options=options(args))
    def save_hook(models, weights, directory):
        if accelerator.is_main_process:
            os.makedirs(directory, exist_ok=True)
            if hasattr(args, '_svm_class_counts'):
                saved['svm_class_counts'] = args._svm_class_counts
            with open(os.path.join(directory, 'shape_options.json'), 'w') as handle:
                json.dump(saved, handle, indent=2)
    accelerator.register_save_state_pre_hook(save_hook)


def count_training_classes(dataset, point_cloud_range):
    """Count labels from training metadata only, matching the object range filter."""
    import numpy as np
    if hasattr(dataset, 'datasets'):
        return sum((count_training_classes(d, point_cloud_range) for d in dataset.datasets),
                   np.zeros(10, dtype=np.int64))
    if hasattr(dataset, 'dataset'):
        return count_training_classes(dataset.dataset, point_cloud_range)
    if tuple(dataset.CLASSES) != CLASS_NAMES or not hasattr(dataset, 'get_ann_info'):
        raise ValueError('Class balancing expects a nuScenes dataset with get_ann_info')
    counts = np.zeros(10, dtype=np.int64)
    bev_range = [point_cloud_range[i] for i in (0, 1, 3, 4)]
    for index in range(len(dataset)):
        ann = dataset.get_ann_info(index)
        labels = np.asarray(ann['gt_labels_3d'])
        # Exactly the same center-in-BEV range convention as ObjectRangeFilter.
        mask = ann['gt_bboxes_3d'].in_range_bev(bev_range).cpu().numpy()
        selected = labels[mask & (labels >= 0) & (labels < 10)]
        # Empty annotation lists can become float64 arrays in nuScenes.
        # bincount requires integer IDs even when its input is empty.
        if selected.size:
            if not np.all(selected == np.floor(selected)):
                raise ValueError('Training class labels must be integer IDs')
            counts += np.bincount(selected.astype(np.int64), minlength=10)
    if counts.sum() == 0:
        raise ValueError('No valid training objects for class-balanced SVM loss')
    return counts


def configure_class_balance(bev_model, dataset, args, point_cloud_range):
    if not (args.use_svm and args.svm_class_balance):
        return None
    model = getattr(bev_model, 'module', bev_model)
    svm = model.pts_bbox_head.shape_svm
    if getattr(args, '_shape_exact', False):
        if svm.class_counts.sum().item() == 0:
            raise ValueError('Class-balanced checkpoint has no saved training class counts')
    else:
        svm.set_class_counts(count_training_classes(dataset, point_cloud_range))
    args._svm_class_counts = svm.class_counts.detach().cpu().tolist()
    return dict(zip(CLASS_NAMES, args._svm_class_counts))
