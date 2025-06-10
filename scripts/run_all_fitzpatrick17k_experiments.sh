#!/bin/bash

# Script to run focused experiments for Fitzpatrick17k:
# 1. Plain Augmix (Torchvision)
# 2. Plain MedMNISTC (Random single corruption from dermamnist, random severity)

# --- Common Settings ---
DATASET_NAME="fitzpatrick17k"
WILDS_ROOT_DIR="../../data/finalfitz17k" 
GPU_ID=0
EPOCHS=50
BATCH_SIZE=128 
LEARNING_RATE=5e-4
WANDB_MODE="online" 
FEAT_DIM=128
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

# HypO specific (used when loss_type is hypo)
PROTO_M=0.95
LOSS_SCALE_W=2.0 

SEEDS=(0 1 2) 
MODELS=("resnet50" "densenet121")
LOSS_TYPES=("erm" "hypo") # Run both ERM and HypO for these augmentations

# Augmentation Configurations
AUG_CONFIGS_NAMES=()
AUG_CONFIGS_PARAMS=()

# Config 1: Plain Torchvision AugMix
# Uses defaults from train_hypo_fitzpatrick.py for tv_augmix_severity, tv_augmix_mixture_width, tv_augmix_alpha
AUG_CONFIGS_NAMES+=("plain_tv_augmix") 
AUG_CONFIGS_PARAMS+=("--use_torchvision_augmix") 

# Config 2: Plain MedMNISTC (Random single corruption from dermamnist, random severity)
# Relies on defaults in fitzpatrick17k_dataloaders_utils.py for random selection logic
AUG_CONFIGS_NAMES+=("plain_medmnistc_random")
AUG_CONFIGS_PARAMS+=("--use_plain_medmnistc")


PYTHON_SCRIPT_PATH="../train_hypo_fitzpatrick.py"

# --- Experiment Loop ---
for model_arch in "${MODELS[@]}"; do
  for loss_type in "${LOSS_TYPES[@]}"; do
    for aug_idx in "${!AUG_CONFIGS_NAMES[@]}"; do
      aug_name=${AUG_CONFIGS_NAMES[$aug_idx]}
      aug_params=${AUG_CONFIGS_PARAMS[$aug_idx]}

      for seed_val in "${SEEDS[@]}"; do
        echo "---------------------------------------------------------------------"
        echo "Running Experiment:"
        echo "Model: ${model_arch}, Loss: ${loss_type}, Augmentation: ${aug_name}, Seed: ${seed_val}"
        echo "---------------------------------------------------------------------"

        TRIAL_NAME="${model_arch}_${loss_type}_${aug_name}_lp${LABEL_PARTITION}_ood${TARGET_DOMAIN_OOD_TEST}_ep${EPOCHS}_seed${seed_val}"
        if [ "$MODEL_PRETRAINED" = true ]; then
            TRIAL_NAME+="_pt" 
        else
            TRIAL_NAME+="_scratch"
        fi

        CMD="python ${PYTHON_SCRIPT_PATH} \\
          --in-dataset ${DATASET_NAME} \\
          --wilds_root_dir ${WILDS_ROOT_DIR} \\
          --model ${model_arch} \\
          --head ${HEAD} \\
          --loss ${loss_type} \\
          --gpu ${GPU_ID} \\
          --epochs ${EPOCHS} \\
          --batch_size ${BATCH_SIZE} \\
          --learning_rate ${LEARNING_RATE} \\
          --lr_decay_epochs \"${LR_DECAY_EPOCHS}\" \\
          --lr_decay_rate ${LR_DECAY_RATE} \\
          --weight-decay ${WEIGHT_DECAY} \\
          --momentum ${MOMENTUM} \\
          --prefetch ${PREFETCH} \\
          --feat_dim ${FEAT_DIM} \\
          --temp ${TEMP} \\
          --trial \"${TRIAL_NAME}\" \\
          --seed ${seed_val} \\
          --model_pretrained ${MODEL_PRETRAINED} \\
          --label_partition ${LABEL_PARTITION} \\
          --target_domain ${TARGET_DOMAIN_OOD_TEST} \\
          --mode ${WANDB_MODE}"

        if [ "$loss_type" = "hypo" ]; then
          CMD+=" \\
          --proto_m ${PROTO_M} \\
          --w ${LOSS_SCALE_W}"
        fi
        
        # Add specific augmentation parameters for the current configuration
        if [ -n "$aug_params" ]; then
            CMD+=" ${aug_params}"
        fi
        
        # Ensure --augment (for basic aug) is not passed if other augs are active
        # The train_hypo_fitzpatrick.py script defaults args.augment to False.
        # The dataloader fitzpatrick17k_dataloaders_utils.py prioritizes 
        # use_plain_medmnistc, then use_torchvision_augmix, then use_med_augmix, then augment_train.
        # So, no explicit --no-augment is needed here if one of the main aug flags is passed.
        # If aug_params is empty (e.g. for a true baseline without any aug flags), 
        # and we wanted basic augmentations, we would add --augment here.
        # But for the requested "plain_tv_augmix" and "plain_medmnistc_random", 
        # aug_params will contain their respective flags, and args.augment will remain False.

        echo "Executing Command:"
        echo "${CMD}"
        echo "---------------------------------------------------------------------"
        
        eval "${CMD}"
        
        echo "Finished: Model: ${model_arch}, Loss: ${loss_type}, Aug: ${aug_name}, Seed: ${seed_val}"
        echo "Log files and checkpoints should be in respective directories under logs/ and checkpoints/ for trial ${TRIAL_NAME}"
        echo "---------------------------------------------------------------------"
        echo ""
        echo ""
      done 
    done 
  done 
done 

echo "Fitzpatrick17k Plain Augmix and Plain MedMNISTC experiments finished."
