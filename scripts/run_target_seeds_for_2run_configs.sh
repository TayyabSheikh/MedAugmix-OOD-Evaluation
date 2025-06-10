#!/bin/bash

# Script to run seeds 0, 1, and 2 for specific configurations
# that were previously reported as having only 2 completed runs.

# --- Common Arguments ---
DATASET="camelyon17"
WILDS_ROOT_DIR="../../data"
GPU_ID=0
EPOCHS=50
BATCH_SIZE=384
LEARNING_RATE=5e-4
WANDB_MODE="online"
FEAT_DIM=128
HEAD="mlp"
TEMP=0.1
LOSS_SCALE_W=2.0 # Default from train_hypo.py, adjust if specific configs used different
PROTO_M=0.95    # Default from train_hypo.py
LR_DECAY_EPOCHS="30,40"
LR_DECAY_RATE=0.1
WEIGHT_DECAY=1e-4
MOMENTUM=0.9
PREFETCH=16

# --- Configurations reported with 2 seeds ---
# base_trial_name MODEL LOSS_TYPE AUG_STRATEGY [AUG_SEV] [AUG_WIDTH]
CONFIGS_TO_RUN=(
    "densenet121_hypo_medmnist_c densenet121 hypo medmnist_c"
    "densenet121_erm_medmnist_c densenet121 erm medmnist_c"
    "resnet50_erm_medmnist_c resnet50 erm medmnist_c"
    "resnet50_hypo_medmnist_c resnet50 hypo medmnist_c"
)

# --- Seeds to run for these configurations ---
SEEDS_TO_RUN=(0 1 2)

echo "Starting targeted seed runs (0, 1, 2) for configurations previously with 2 runs..."

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

for config_line in "${CONFIGS_TO_RUN[@]}"; do
    IFS=' ' read -r base_trial_name MODEL LOSS_TYPE AUG_STRATEGY AUG_SEV AUG_WIDTH <<< "$config_line"

    echo "" # Add a blank line for better readability between configurations
    echo "# --- Processing Configuration: ${base_trial_name} ---"

    PYTHON_SCRIPT=""
    AUG_FLAGS=""

    case "$AUG_STRATEGY" in
        baseline)
            # This script is not targeting baseline, but keeping for structure
            PYTHON_SCRIPT="../train_hypo.py" # Handles ERM via --loss
            ;;
        medmnist_c)
            if [ "$LOSS_TYPE" == "erm" ]; then
                PYTHON_SCRIPT="../train_erm_medmnistc.py"
            else # hypo
                PYTHON_SCRIPT="../train_hypo_medmnistc.py"
            fi
            # No extra flags needed if these scripts inherently apply basic MedMNIST-C
            ;;
        plain_augmix)
            # This script is not targeting plain_augmix, but keeping for structure
            PYTHON_SCRIPT="../train_plain_augmix.py"
            ;;
        medmnist_c_augmix)
            # This script is not targeting medmnist_c_augmix, but keeping for structure
            if [ "$LOSS_TYPE" == "erm" ]; then
                PYTHON_SCRIPT="../train_erm_medmnistc.py"
            else # hypo
                PYTHON_SCRIPT="../train_hypo_medmnistc.py"
            fi
            AUG_FLAGS="--use_med_augmix --augmix_severity ${AUG_SEV} --augmix_mixture_width ${AUG_WIDTH}"
            ;;
        *)
            echo "# Unknown augmentation strategy: $AUG_STRATEGY for $base_trial_name. Skipping."
            continue
            ;;
    esac
    
    if [ -z "$PYTHON_SCRIPT" ]; then
        echo "# Could not determine Python script for $base_trial_name. Skipping."
        continue
    fi

    for SEED in "${SEEDS_TO_RUN[@]}"; do
        TRIAL="${base_trial_name}_seed_${SEED}" # Construct full trial name
        
        echo "Running: Model=${MODEL}, Loss=${LOSS_TYPE}, Aug=${AUG_STRATEGY}, Seed=${SEED}"
        echo "Trial Name: ${TRIAL}"
        echo "Python Script: ${PYTHON_SCRIPT}"

        CMD="python ${SCRIPT_DIR}/${PYTHON_SCRIPT} \\
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
          --mode ${WANDB_MODE} \\
          ${AUG_FLAGS}"
        
        echo "Command:"
        echo -e "${CMD//\\$'\n'/}" # Print command in a more readable single line for logs
        echo ""

        eval "$CMD"

        echo "Finished run for: ${TRIAL}"
        echo "----------------------------------------"
    done
    echo "# ---------------------------------------- (End of commands for ${base_trial_name})"
done

echo ""
echo "All targeted seed runs initiated."
