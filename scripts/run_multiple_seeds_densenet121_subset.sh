#!/bin/bash

# Script to run ERM, HypO, HypO+MedC, and ERM+MedC training on DenseNet121
# for 50 epochs with multiple random seeds.

# Activate conda environment (adjust if your conda init is different)
# source $(conda info --base)/etc/profile.d/conda.sh # Common way, might vary
# conda activate fresh # Assuming 'fresh' is the environment name

# --- Common Arguments ---
DATASET="camelyon17"
WILDS_ROOT_DIR="../../data" # Relative path from scripts/ to data/
MODEL="resnet50"
GPU_ID=0 # Adjust GPU ID as needed
EPOCHS=50
BATCH_SIZE=384
LEARNING_RATE=5e-4
WANDB_MODE="online" # Set to "disabled" to turn off wandb logging
FEAT_DIM=128
HEAD="mlp"
TEMP=0.1
LOSS_SCALE_W=2.0
PROTO_M=0.95
LR_DECAY_EPOCHS="30,40" # Adjusted for 50 epochs
LR_DECAY_RATE=0.1
WEIGHT_DECAY=1e-4
MOMENTUM=0.9
PREFETCH=16

# --- Seeds to run ---
SEEDS=(0 1 2) # Run with seeds 0, 1, and 2

# --- Training Configurations ---
# Format: LOSS_TYPE PYTHON_SCRIPT_NAME TRIAL_PREFIX
CONFIGS=(
    "erm train_hypo.py erm"
    "hypo train_hypo.py hypo"
    "hypo train_hypo_medmnistc.py hypo_medmnistc"
    "erm train_erm_medmnistc.py erm_medmnistc"
)

echo "Starting multiple seed training runs for DenseNet121 (Subset of Configs)..."

# Loop through each configuration
for config in "${CONFIGS[@]}"; do
    read -r LOSS_TYPE PYTHON_SCRIPT TRIAL_PREFIX <<< "$config"

    # Loop through each seed
    for SEED in "${SEEDS[@]}"; do
        TRIAL="${TRIAL_PREFIX}_seed_${SEED}" # Unique trial name including seed

        echo "Running ${LOSS_TYPE} (${TRIAL_PREFIX}) with seed ${SEED}..."
        echo "Trial Name: ${TRIAL}"

        # Construct the command
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

        # Print the command
        echo "Command:"
        echo -e "$CMD\n" # Use -e to interpret escape sequences like newline

        # Execute the command
        # Note: Running in background with & might be useful for multiple GPUs
        # For sequential execution, remove '&' and 'wait'
        eval "$CMD" # Use eval to execute the command string with newlines

        echo "${LOSS_TYPE} (${TRIAL_PREFIX}) with seed ${SEED} finished."
        echo "----------------------------------------"
    done
done

echo "All specified training runs completed."
