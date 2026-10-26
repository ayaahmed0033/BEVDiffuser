# Confidence-Gated Temporal Cross-Attention for BEVDiffuser

This is an optional add-on for BEVDiffuser. It leaves the existing SVM, bottleneck, detector, LiDAR paths, input augmentation, and configuration files in place.

## What it adds

During BEVFormer’s normal temporal processing, the installer captures the previous-frame `prev_bev` feature. At the diffusion U-Net middle block, the current noisy BEV feature queries that previous BEV map through multi-head cross-attention. A learned per-location confidence gate scales the attention residual so the U-Net can reduce the contribution of inconsistent history. The residual projection starts at zero, so enabling the module begins as an identity path and it learns to use temporal context during training.

When BEVFormer has no previous feature (for example, at a new scene boundary), the attention path is bypassed for that sample. The captured context is detached from the detector graph. The diffusion model remains responsible for denoising; the new module does not replace the diffusion process or add a classifier.

The feature is optional and is active only when explicitly enabled. It adds trainable parameters to the U-Net, so they are included in optimization and the normal U-Net checkpoint. A small JSON file beside each checkpoint stores its attention settings. Evaluation and checkpoint resume restore those settings automatically.

To keep runs separate, the trainer adds `-temporal` to the output directory and run name when the flag is active. Existing shape-experiment suffixes are retained.

## Install

Extract this ZIP, then run:

```bash
conda activate bevdiffuser
python ~/BEVDiffuser_TemporalCrossAttention/install_temporal_attention.py \
  --repo ~/BEVDiffuser --apply
```

The installer first validates every source change, checks for active Git conflicts, refuses to replace unknown existing files, and creates a timestamped backup before editing. It changes only `train_bev_diffuser.py`, `test_bev_diffuser.py`, `train.sh`, `test.sh`, and adds `temporal_cross_attention.py`.

## Train

From `~/BEVDiffuser/BEVFormer/projects/bevdiffuser`:

```bash
CUDA_VISIBLE_DEVICES=0 bash train.sh 1 --use_temporal_cross_attention
```

Optional tuning flags:

```bash
--temporal_attention_heads 4
--temporal_attention_dropout 0.0
```

They default to 4 heads and no attention dropout. Existing options such as the SVM, bottleneck, class balancing, checkpoint, and training-step arguments can be supplied on the same command line.

## Evaluate

```bash
CUDA_VISIBLE_DEVICES=0 bash test.sh 1
```

The evaluator reads the checkpoint metadata and constructs the same temporal module before loading its weights. If the selected checkpoint was not trained with this module, the evaluator leaves it disabled. A temporal checkpoint missing its metadata or temporal weights raises an error instead of silently evaluating with random parameters.

The package includes CPU unit tests for the attention residual, confidence gradients, previous-BEV capture, and scene-start bypass. Run them in an environment with the project’s PyTorch installed:

```bash
PYTHONPATH=files/BEVFormer/projects/bevdiffuser \
  python -m unittest discover -s tests -p 'test_temporal_cross_attention.py'
```

## Notes and limits

- The attention is applied at the U-Net bottleneck on every U-Net forward. Training uses one forward per sampled diffusion noise level; inference uses one per reverse-denoising iteration.
- Confidence is a learned gate, not a calibrated probability or an external reliability label.
- BEVFormer’s previous feature is already rotated through its own temporal path. This add-on does not duplicate or modify BEVFormer’s detector-side temporal attention, pose/timestamp handling, or translation-shift calculation.
- The installer does not launch training or alter the LiDAR pipeline. Run a small training check in the `bevdiffuser` environment before a full experiment.
