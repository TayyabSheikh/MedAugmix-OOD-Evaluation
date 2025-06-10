#!/bin/bash

# Script to train HypO on Fitzpatrick17k

# Define arguments
DATASET="fitzpatrick17k"
WILDS_ROOT_DIR="../../data/finalfitz17k"
MODEL="densenet121" # Or resnet50, etc.
GPU_ID=0
EPOCHS=50 # As per train_hypo_fitzpatrick.py default
BATCH_SIZE=32 # As per train_hypo_fitzpatrick.py default
LEARNING_RATE=5e-4 # As per train_hypo_fitzpatrick.py default
WANDB_MODE="online" # Set to "disabled" to turn off wandb logging
FEAT_DIM=128 # As per train_hypo_fitzpatrick.py default
LOSS_TYPE="hypo"
TRIAL="hypo_baseline_0" # Increment for multiple runs
HEAD="mlp" # As per train_hypo_fitzpatrick.py default
WEIGHT_DECAY=1e-4 # As per train_hypo_fitzpatrick.py default
MOMENTUM=0.9 # As per train_hypo_fitzpatrick.py default
PREFETCH=4 # As per train_hypo_fitzpatrick.py default
TEMP=0.1 # As per train_hypo_fitzpatrick.py default
LR_DECAY_EPOCHS="30,40" # As per train_hypo_fitzpatrick.py default
LR_DECAY_RATE=0.1 # As per train_hypo_fitzpatrick.py default
MODEL_PRETRAINED=True # As per train_hypo_fitzpatrick.py default

# HypO specific parameters
PROTO_M=0.95 # As per train_hypo_fitzpatrick.py default
LOSS_SCALE_W=2 # As per train_hypo_fitzpatrick.py default (w)

# Fitzpatrick17k specific
LABEL_PARTITION=3
TARGET_DOMAIN_OOD_TEST="56" # FST group for OOD test

# Augmentation setting for baseline (no augmentation)
AUGMENT_TRAIN=False # Explicitly set to False

# Construct the command
# Assumes the script is run from the hypo_impl/scripts directory

echo "Running HypO training script for Fitzpatrick17k from scripts directory..."
echo "Training with NO augmentations."
echo "Command:"
echo "python ../train_hypo_fitzpatrick.py \\"
echo "  --in-dataset ${DATASET} \\"
echo "  --wilds_root_dir ${WILDS_ROOT_DIR} \\"
echo "  --model ${MODEL} \\"
echo "  --head ${HEAD} \\"
echo "  --loss ${LOSS_TYPE} \\"
echo "  --gpu ${GPU_ID} \\"
echo "  --epochs ${EPOCHS} \\"
echo "  --batch_size ${BATCH_SIZE} \\"
echo "  --learning_rate ${LEARNING_RATE} \\"
echo "  --lr_decay_epochs ${LR_DECAY_EPOCHS} \\"
echo "  --lr_decay_rate ${LR_DECAY_RATE} \\"
echo "  --weight-decay ${WEIGHT_DECAY} \\"
echo "  --momentum ${MOMENTUM} \\"
echo "  --prefetch ${PREFETCH} \\"
echo "  --feat_dim ${FEAT_DIM} \\"
echo "  --proto_m ${PROTO_M} \\"
echo "  --w ${LOSS_SCALE_W} \\"
echo "  --temp ${TEMP} \\"
echo "  --trial ${TRIAL} \\"
echo "  --mode ${WANDB_MODE} \\"
echo "  --model_pretrained ${MODEL_PRETRAINED} \\"
echo "  --label_partition ${LABEL_PARTITION} \\"
echo "  --target_domain ${TARGET_DOMAIN_OOD_TEST} \\"
echo "  --augment ${AUGMENT_TRAIN} \\"
echo "  --use_med_augmix False"

# Execute the command
python ../train_hypo_fitzpatrick.py \
  --in-dataset ${DATASET} \
  --wilds_root_dir ${WILDS_ROOT_DIR} \
  --model ${MODEL} \
  --head ${HEAD} \
  --loss ${LOSS_TYPE} \
  --gpu ${GPU_ID} \
  --epochs ${EPOCHS} \
  --batch_size ${BATCH_SIZE} \
  --learning_rate ${LEARNING_RATE} \
  --lr_decay_epochs ${LR_DECAY_EPOCHS} \
  --lr_decay_rate ${LR_DECAY_RATE} \
  --weight-decay ${WEIGHT_DECAY} \
  --momentum ${MOMENTUM} \
  --prefetch ${PREFETCH} \
  --feat_dim ${FEAT_DIM} \
  --proto_m ${PROTO_M} \
  --w ${LOSS_SCALE_W} \
  --temp ${TEMP} \
  --trial ${TRIAL} \
  --mode ${WANDB_MODE} \
  --model_pretrained ${MODEL_PRETRAINED} \
  --label_partition ${LABEL_PARTITION} \
  --target_domain ${TARGET_DOMAIN_OOD_TEST} \
  --augment ${AUGMENT_TRAIN} \
  --use_med_augmix False

echo "HypO training script for Fitzpatrick17k finished."
