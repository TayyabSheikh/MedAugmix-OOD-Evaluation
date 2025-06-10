#!/bin/bash

# Script to run HypO and ERM with PLAIN AugMix on ResNet50
# for 50 epochs with multiple random seeds.

# --- Common Arguments ---
DATASET="camelyon17"
WILDS_ROOT_DIR="../../data" # Relative path from scripts/ to data/
MODEL="resnet50"
GPU_ID=0
EPOCHS=50
BATCH_SIZE=384
LEARNING_RATE=5e-4
WANDB_MODE="online"
FEAT_DIM=128
HEAD="mlp"
TEMP=0.1
LOSS_SCALE_W=2.0
PROTO_M=0.95
LR_DECAY_EPOCHS="30,40"
LR_DECAY_RATE=0.1
WEIGHT_DECAY=1e-4
MOMENTUM=0.9
PREFETCH=16

# --- Seeds to run ---
SEEDS=(0 1 2)

# --- Training Configurations ---
# Format: LOSS_TYPE PYTHON_SCRIPT_NAME TRIAL_BASE_PREFIX
CONFIGS=(
    "hypo train_plain_augmix.py resnet50_hypo_plain_augmix"
    "erm train_plain_augmix.py resnet50_erm_plain_augmix"
)

echo "Starting multiple seed training runs with plain AugMix for ResNet50..."

for config in "${CONFIGS[@]}"; do
    read -r LOSS_TYPE PYTHON_SCRIPT TRIAL_BASE_PREFIX <<< "$config"

    for SEED in "${SEEDS[@]}"; do
        TRIAL="${TRIAL_BASE_PREFIX}_seed_${SEED}"

        echo "Running ${LOSS_TYPE} (${TRIAL_BASE_PREFIX}) with Seed ${SEED}..."
        echo "Trial Name: ${TRIAL}"

        CMD="python ../${PYTHON_SCRIPT} \\
          --in-dataset ${DATASET} \\
          --wilds_root_dir ${WILDS_ROOT_DIR} \\
          --model ${MODEL} \\
          --head ${HEAD} \\
          --loss ${LOSS_TYPE} \\
          --gpu ${GPU_ID} \\
          --epochs ${EPOCHS} \\
          --batch_size ${BATCH_SIZE} \\
          --learning_rate ${LEARNING_RATE} \\
          --lr_decay_epochs ${LR_DECAY_EPOCHS} \\
          --lr_decay_rate ${LR_DECAY_RATE} \\
          --weight-decay ${WEIGHT_DECAY} \\
          --momentum ${MOMENTUM} \\
          --prefetch ${PREFETCH} \\
          --feat_dim ${FEAT_DIM} \\
          --temp ${TEMP} \\
          --w ${LOSS_SCALE_W} \\
          --proto_m ${PROTO_M} \\
          --trial ${TRIAL} \\
          --seed ${SEED} \\
          --mode ${WANDB_MODE}"
        
        # Note: No --use_plain_augmix flag needed as train_plain_augmix.py applies it by default for Camelyon17.
        # Ensure no --use_med_augmix or related flags are passed if they are not intended.

        echo "Command:"
        echo -e "$CMD\n"

        eval "$CMD"

        echo "${LOSS_TYPE} (${TRIAL_BASE_PREFIX}) with seed ${SEED} finished."
        echo "----------------------------------------"
    done
done

echo "All specified plain AugMix training runs for ResNet50 completed."
