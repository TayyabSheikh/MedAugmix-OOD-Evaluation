#!/bin/bash

# Script to train ERM on Fitzpatrick17k with general MedAugMix

# Define arguments
DATASET="fitzpatrick17k"
WILDS_ROOT_DIR="../../data/finalfitz17k"
MODEL="densenet121" # Or resnet50, etc.
GPU_ID=0
EPOCHS=50
BATCH_SIZE=32
LEARNING_RATE=5e-4
WANDB_MODE="online"
FEAT_DIM=128
LOSS_TYPE="erm"
TRIAL="erm_medaugmix_configurable_0" # Specific trial name
HEAD="mlp"
WEIGHT_DECAY=1e-4
MOMENTUM=0.9
PREFETCH=4
TEMP=0.1
LR_DECAY_EPOCHS="30,40"
LR_DECAY_RATE=0.1
MODEL_PRETRAINED=True

# Fitzpatrick17k specific
LABEL_PARTITION=3
TARGET_DOMAIN_OOD_TEST="56"

# MedAugMix specific settings (configurable)
USE_MED_AUGMIX=True
AUGMIX_CORRUPTION_DATASET="dermamnist" # Default, can be changed in this script
AUGMIX_SEVERITY=3 # Default, can be changed in this script
AUGMIX_MIXTURE_WIDTH=3 # Default, can be changed in this script
# The --augment flag in train_hypo_fitzpatrick.py is effectively ignored by the dataloader
# when use_med_augmix is True, due to the dataloader modification.
# We can pass --augment False for clarity, though it won't change behavior with MedAugMix.
AUGMENT_TRAIN=False

# Construct the command
echo "Running ERM training script for Fitzpatrick17k with MedAugMix..."
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
echo "  --temp ${TEMP} \\"
echo "  --trial ${TRIAL} \\"
echo "  --mode ${WANDB_MODE} \\"
echo "  --model_pretrained ${MODEL_PRETRAINED} \\"
echo "  --label_partition ${LABEL_PARTITION} \\"
echo "  --target_domain ${TARGET_DOMAIN_OOD_TEST} \\"
echo "  --augment ${AUGMENT_TRAIN} \\" # For clarity, though overridden by use_med_augmix logic in dataloader
echo "  --use_med_augmix ${USE_MED_AUGMIX} \\"
echo "  --augmix_corruption_dataset ${AUGMIX_CORRUPTION_DATASET} \\"
echo "  --augmix_severity ${AUGMIX_SEVERITY} \\"
echo "  --augmix_mixture_width ${AUGMIX_MIXTURE_WIDTH}"

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
  --temp ${TEMP} \
  --trial ${TRIAL} \
  --mode ${WANDB_MODE} \
  --model_pretrained ${MODEL_PRETRAINED} \
  --label_partition ${LABEL_PARTITION} \
  --target_domain ${TARGET_DOMAIN_OOD_TEST} \
  --augment ${AUGMENT_TRAIN} \
  --use_med_augmix ${USE_MED_AUGMIX} \
  --augmix_corruption_dataset ${AUGMIX_CORRUPTION_DATASET} \
  --augmix_severity ${AUGMIX_SEVERITY} \
  --augmix_mixture_width ${AUGMIX_MIXTURE_WIDTH}

echo "ERM training script for Fitzpatrick17k with MedAugMix finished."
