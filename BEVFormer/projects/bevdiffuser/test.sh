#!/usr/bin/env bash
# TEMPORAL CONFIDENCE ATTENTION integration v1
# SHAPE SVM optional integration v2
# EVSSM optional cached camera restoration
EVSSM_TEST_GPUS=1
if [[ ${1:-} =~ ^[0-9]+$ ]]; then EVSSM_TEST_GPUS=$1; shift; fi
set -e

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

BEV_CONFIG="../configs/bevdiffuser/layout_tiny.py"

CHECKPOINT_DIR="../../train/svm_rbf-shape-temporal/checkpoint-50000"

BEV_CHECKPOINT="../../train/svm_rbf-shape-temporal/checkpoint-50000/bev_model.pth"
# "../../ckpts/bevformer_tiny_epoch_24.pth" 

PREDICTION_TYPE="sample"

python -m torch.distributed.launch --nproc_per_node="$EVSSM_TEST_GPUS" --master_port 9999 test_bev_diffuser.py \
    --bev_config $BEV_CONFIG \
    --bev_checkpoint $BEV_CHECKPOINT \
    --checkpoint_dir $CHECKPOINT_DIR \
    --prediction_type $PREDICTION_TYPE \
    --noise_timesteps 5 \
    --denoise_timesteps 5 \
    --num_inference_steps 5 \
    --use_unet_bottleneck \
    --bottleneck_time_gate \
    --use_svm \
    --svm_kernel rbf \
    --svm_rbf_features 512 \
    --svm_class_balance \
    --use_temporal_cross_attention \
    "$@"

    # --use_classifier_guidence \
    #--use_svm \