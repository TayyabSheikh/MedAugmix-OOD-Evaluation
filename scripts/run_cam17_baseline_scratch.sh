#!/bin/bash

# Script to run baseline ERM and HypO experiments on Camelyon17
# with ResNet50 and DenseNet121, training from scratch (no pretrained weights).
# Runs for a single seed (0).

# --- Common Arguments ---
DATASET="camelyon17"
WILDS_ROOT_DIR="../../data"
GPU_ID=0
EPOCHS=50
BATCH_SIZE=384 # Adjusted to common batch size, can be changed if needed
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
PREFETCH=16 # Adjusted to common prefetch, can be changed

# --- Models and Loss Types to Run ---
MODELS_TO_RUN=("resnet50" "densenet121")
LOSS_TYPES=("erm" "hypo")
SEED_TO_RUN=0

echo "Starting Camelyon17 baseline experiments (training from scratch)..."

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PYTHON_SCRIPT_PATH="${SCRIPT_DIR}/../train_hypo.py"

for MODEL_NAME in "${MODELS_TO_RUN[@]}"; do
    for LOSS_TYPE in "${LOSS_TYPES[@]}"; do
        
        TRIAL_NAME="${MODEL_NAME}_${LOSS_TYPE}_baseline_scratch_seed_${SEED_TO_RUN}"

        echo ""
        echo "# --- Running Configuration ---"
        echo "# Model: ${MODEL_NAME}"
        echo "# Loss Type: ${LOSS_TYPE}"
        echo "# Seed: ${SEED_TO_RUN}"
        echo "# Pretrained: False"
        echo "# Trial Name: ${TRIAL_NAME}"
        echo "# Python Script: ${PYTHON_SCRIPT_PATH}"
        echo "# ---------------------------"

        CMD="python ${PYTHON_SCRIPT_PATH} \\
          --in-dataset ${DATASET} \\
          --wilds_root_dir ${WILDS_ROOT_DIR} \\
          --model ${MODEL_NAME} \\
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
          --trial ${TRIAL_NAME} \\
          --seed ${SEED_TO_RUN} \\
          --model_pretrained False \\
          --mode ${WANDB_MODE}"
        
        echo "Command:"
        echo -e "${CMD//\\$'\n'/}" # Print command in a more readable single line for logs
        echo ""

        eval "$CMD"

        echo "Finished run for: ${TRIAL_NAME}"
        echo "----------------------------------------"
    done
done

echo ""
echo "All baseline scratch experiments initiated."
