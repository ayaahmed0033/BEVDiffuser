# BEVDiffuser RBF shape-SVM experiment

Includes the selected additions: timestep-conditioned bottleneck gates and
class-balanced SVM loss. RGB/grayscale consistency loss is not included.
This is independently implemented experimental code, not a reproduction of
CBGS, SSN or an existing published BEVDiffuser/SVM method.

## Install

Extract this folder inside `/home/aya/BEVDiffuser`, then run:

```bash
python /home/aya/BEVDiffuser/BEVDiffuser_RBF_SVM/install_shape_svm.py \
  --repo /home/aya/BEVDiffuser --apply
```

No Torch, CUDA, MMCV, conda or scikit-learn installation is performed. The
installer uses standard-library Python and patches the existing Python entry
points incrementally. If the earlier ZIP's exact files are installed, it safely
upgrades those files; locally modified versions cause a clear stop for review.
It recognizes both exact v2 shape helpers: the original helper and the version
with the empty-class-count guard. It supports the restored baseline and the
optional RAF integration, including the installed optional EVSSM cache hooks. It preserves
your paths, launcher hyperparameters and existing unrelated code. It backs up
changed files before writing, rolls back on write failure, and is idempotent.
An incompatible integration anchor or active Git merge/rebase stops the
installation before any file changes. No commits, pulls or pushes are made.

The launchers forward extra arguments and consume the GPU-count argument.
Standalone hardcoded `--use_raf`/`--no_raf` lines are removed so RAF is chosen
explicitly with `--use_raf` at invocation. Existing Python RAF/EVSSM modules
remain in place; neither feature is enabled by the new experiment flags.
No dataset converter, LiDAR loader, sensor modality setting, normalization,
calibration, scheduler or original U-Net source file is replaced.

## Two-step smoke run

```bash
conda activate bevdiffuser
cd /home/aya/BEVDiffuser/BEVFormer/projects/bevdiffuser

CUDA_VISIBLE_DEVICES=0 bash train.sh 1 \
  --use_svm --svm_kernel rbf --grayscale_aug --use_unet_bottleneck \
  --bottleneck_time_gate --svm_class_balance \
  --output_dir ../../train/SHAPE_SVM_smoke \
  --tracker_run_name SHAPE_SVM_smoke \
  --max_train_steps 2 \
  --checkpointing_steps 2 \
  --dataloader_num_workers 0 \
  --skip_checkpoint_eval
```

Expect `Total optimization steps = 2`, a log of training annotation class counts,
and `Installed 19 residual U-Net bottlenecks`. Class counting adds a metadata
pass at startup; it does not load images, use validation labels, or train a
separate offline classifier. The checkpoint is:

```text
/home/aya/BEVDiffuser/BEVFormer/train/SHAPE_SVM_smoke-shape/checkpoint-2
```

## Full training

```bash
CUDA_VISIBLE_DEVICES=0 bash train.sh 1 \
  --use_svm --svm_kernel rbf --grayscale_aug --use_unet_bottleneck \
  --bottleneck_time_gate --svm_class_balance \
  --output_dir ../../train/SHAPE_SVM \
  --tracker_run_name SHAPE_SVM \
  --max_train_steps 50100 \
  --checkpointing_steps 50000
```

Expected checkpoint:

```text
/home/aya/BEVDiffuser/BEVFormer/train/SHAPE_SVM-shape/checkpoint-50000
```

To combine with the already-installed RAF integration, add `--use_raf`. The
output then has a `-shape-raf` suffix. RAF has its own checkpoint identity and
must also be enabled when evaluating a combined checkpoint. The pure shape
experiment requires no RAF or EVSSM weights.

## Evaluate

```bash
CUDA_VISIBLE_DEVICES=0 bash test.sh 1 \
  --checkpoint_dir ../../train/SHAPE_SVM-shape/checkpoint-50000 \
  --eval bbox
```

The checkpoint's `shape_options.json` reconstructs its trained SVM and adapter
configuration automatically, including gates. Its saved `bev_model.pth` is
used for the trained SVM. Random grayscale and class balancing are training
operations only; the saved SVMs and gated adapters also run at evaluation.
Earlier v2 shape-SVM checkpoints are restored in linear mode so their saved
classifier weights keep the correct dimensions.
The launcher must use the same BEV config as training. Old baseline/RAF
checkpoints cannot supply trained SVMs or gated adapters: initialize a new
training run from baseline weights rather than treating an old checkpoint as
the new method.

## Flags and initial settings

| Flag | Default | Effect |
| --- | --- | --- |
| `--use_svm` | off | Fine/coarse max-margin SVM heads, trained jointly in PyTorch |
| `--svm_kernel` | `rbf` | Differentiable RBF-kernel approximation; `linear` preserves old checkpoints |
| `--svm_rbf_features` | 512 | Fixed random Fourier features used to approximate the RBF kernel |
| `--svm_rbf_gamma` | automatic | RBF width; defaults to `1 / descriptor_dimension` |
| `--grayscale_aug` | off | Random three-channel grayscale before normalization |
| `--grayscale_prob` | 0.3 | Grayscale probability during training |
| `--use_unet_bottleneck` | off | Residual adapters after 19 complete tiny U-Net blocks |
| `--bottleneck_ratio` | 4 | C -> C/4 -> C/4 -> C within each adapter |
| `--bottleneck_time_gate` | off | Learned sigmoid gate from the existing timestep/layout embedding |
| `--svm_class_balance` | off | Training-frequency weights for positive fine/coarse SVM loss |
| `--svm_fusion_weight` | 0.1 | Bounded margin correction to existing detection logits |
| `--svm_loss_weight` | 0.1 | SVM objective weight inside the existing task loss |
| `--svm_l2_weight` | 0.0001 | Explicit L2 penalty on SVM weights |

Omit optional flags to disable their respective additions. The gate requires
the bottleneck flag; class balancing requires the SVM flag. SVM training
requires `--prediction_type sample` and a positive `--task_loss_scale`.

## What the method does

The SVM descriptor contains normalized object decoder features, predicted
log width/length/height and three log dimension ratios. Fine classification
has ten classes plus background. The coarse groups are car; large vehicles
(truck, construction vehicle, bus, trailer); barrier; two-wheelers; pedestrian;
traffic cone; plus background. Groups provide extra supervision without
merging final labels. Features can contain color from RGB training examples;
grayscale augmentation encourages less color dependence, not color-free
classification. There is no HOG extractor, silhouette segmentation or separate
edge-restoration module in this package.

The default RBF kernel is approximated with fixed random Fourier features,
then trained with multiclass hinge loss plus explicit L2. This keeps the RBF
mapping differentiable so gradients can flow back through the detection head
and into BEV denoising. It is not an offline scikit-learn SVC. Use
`--svm_kernel linear` to reproduce the earlier primal linear version. Existing
focal and box regression losses remain. Up to 128 hard background queries
are included with lower weight. Foreground-minus-background margins correct
the original class scores; these margins are not calibrated probabilities.
Fine and coarse SVM weights start at zero, leaving scores unchanged at
initialization.

### SVM in the diffusion denoising objective

The SVM participates in denoiser training: one training batch samples a noise
timestep, the U-Net predicts the clean BEV (`model_pred`), and the detector plus
SVM evaluate that prediction. The SVM hinge objective is included in the
existing task loss, so the total is the diffusion MSE plus the scaled detector
losses (including the weighted SVM loss). With `--prediction_type sample` and
positive `--task_loss_scale`, gradients flow from the SVM through the detector
to the predicted BEV and back into the U-Net. No forward-noising scheduler or
original U-Net code is changed. This computes the SVM loss once for the sampled
training timestep in each batch; it does not apply SVM guidance at every (or
any) reverse inference step. The baseline's existing classifier-guidance hook
is separate and calls an evaluation-loss path with the full batch, so do not
enable it as SVM guidance for a fair test. Sparse reverse-step SVM guidance
would need a separately defined label-free score so evaluation targets are
never used to steer predictions; this package does not add that mode.

Class weights are computed from training labels after the same BEV range
filter used by the detector. Inverse square-root counts are normalized over
present classes and clipped to [0.25, 4]. Fine and coarse losses have their
own weights. Absent classes receive zero positive weight; background keeps
its separate original weight. Counts and weights are stored as checkpoint
buffers and reused at resume/test. No validation frequencies are used. This
weighting is an experimental choice, not a verified accuracy improvement.

Grayscale conversion retains three BGR channels, camera order and geometry.
All views in a frame share the conversion decision; historical frames sample
independently. Existing color augmentation stays active. No images are saved
or cached by this experiment.

Adapters preserve spatial size, channel size and U-Net skip connections. Their
zero-initialized final projection makes each residual branch initially zero.
When selected, a learned scalar sigmoid gate adjusts each block's residual
using the existing embedding, which combines timestep and layout information.
They run at each sampled training timestep and every inference denoising call.
Forward Gaussian noising remains outside the U-Net in the scheduler. Adapters
add computation and parameters; they do not shrink or speed up the original
U-Net. The camera/BEV extractor retains its baseline freezing policy, while
the SVM loss reaches connected trainable decoder and denoiser parameters.

## Scope and validation

The original baseline's annotated-layout conditioning is preserved, including
its existing evaluation path. This package does not make that pipeline
ground-truth-free autonomous inference. Its SVM descriptor itself receives
predicted geometry rather than ground-truth dimensions.

The CPU test suite covers the actual layout-aware U-Net at smaller channel
width, object-aware attention, gated/ungated identity and gradients, SVM
objective optimization, RBF-kernel approximation and gradients, old-checkpoint
migration, rare-class weighting, grayscale behavior and checkpoint reloads.
The package build's six standard-library installer checks passed, and all Python
files parsed successfully. PyTorch and pytest are not installed in the build
workspace, so the model-forward/backward tests were not executed here. Compiled
MMCV/CUDA distributed training and nuScenes accuracy also remain unverified.

Optional local checks after installing:

```bash
python -m pytest tests/test_shape_experiment.py -q
python BEVFormer/projects/bevdiffuser/train_bev_diffuser.py --help
```

Installer checks also cover baseline, RAF and EVSSM integration variants,
environment-setting preservation, repeated apply, backup creation, argument
forwarding and refusal to partially install incompatible entry points.
