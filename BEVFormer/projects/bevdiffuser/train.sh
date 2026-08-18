#!/usr/bin/env bash

GPUS=$1
PORT=${PORT:-29503}

BEV_CONFIG="../configs/bevdiffuser/layout_tiny.py"
BEV_CHECKPOINT="../../ckpts/bevformer_tiny_epoch_24.pth"
PRETRAINED_MODEL="/home/aya/BEVDiffuser/BEVFormer/hf_models/stable-diffusion-2-1"
PRETRAINED_UNET_CHECKPOINT=None

PROJ_NAME=BEVDiffusers
RUN_NAME=noise_construction

CHECKPOINT_STEP=1000
CHECKPOINT_LIMIT=20

MAX_TRAINING_STEPS=10001
TRAIN_BATCH_SIZE=1
DATALOADER_NUM_WORKERS=8
GRADIENT_ACCUMMULATION_STEPS=8

LEARNING_RATE=1e-4
LR_SCHEDULER="constant"

UNCOND_PROB=0.2
PREDICTION_TYPE="sample"
TASK_LOSS_SCALE=0.1

OUTPUT_DIR="../../train/${RUN_NAME}"

mkdir -p $OUTPUT_DIR

PYTHONPATH="$(dirname $0)/../..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
  $(dirname "$0")/train_bev_diffuser.py \
    --bev_config $BEV_CONFIG \
    --bev_checkpoint $BEV_CHECKPOINT \
    --pretrained_unet_checkpoint $PRETRAINED_UNET_CHECKPOINT \
    --pretrained_model_name_or_path $PRETRAINED_MODEL \
    --train_batch_size $TRAIN_BATCH_SIZE \
    --dataloader_num_workers $DATALOADER_NUM_WORKERS \
    --gradient_accumulation_steps $GRADIENT_ACCUMMULATION_STEPS \
    --max_train_steps $MAX_TRAINING_STEPS \
    --learning_rate $LEARNING_RATE \
    --lr_scheduler $LR_SCHEDULER \
    --output_dir $OUTPUT_DIR \
    --checkpoints_total_limit $CHECKPOINT_LIMIT \
    --checkpointing_steps $CHECKPOINT_STEP \
    --tracker_run_name $RUN_NAME \
    --tracker_project_name $PROJ_NAME \
    --uncond_prob $UNCOND_PROB \
    --prediction_type $PREDICTION_TYPE \
    --task_loss_scale $TASK_LOSS_SCALE \
    --use_ncm   \
    --resume_from_checkpoint /home/aya/BEVDiffuser/BEVFormer/train/noise_construction/checkpoint-2000 
        #--report_to 'wandb'