#!/bin/bash

# Script to run ERM and HypO with single random MedMNIST-C augmentations
# using the modified train_hypo.py script.

# --- Common Arguments ---
DATASET="camelyon17"
WILDS_ROOT_DIR="../../data" 
EPOCHS=50
BATCH_SIZE=384
LEARNING_RATE=5e-4
WANDB_MODE="online" 
FEAT_DIM=128
HEAD="mlp"
TEMP=0.1
LOSS_SCALE_W=2.0 # For HypO
PROTO_M=0.95    # For HypO
LR_DECAY_EPOCHS="30,40"
LR_DECAY_RATE=0.1
WEIGHT_DECAY=1e-4
MOMENTUM=0.9
PREFETCH=16
CORRUPTION_SOURCE="bloodmnist" # Default source for MedMNIST-C functions
GPU_ID=0 # Adjust as needed

# --- Seeds to run ---
SEEDS=(0 1 2)

# --- Models to run ---
MODELS=("resnet50" "densenet121")

# --- Python script to use ---
PYTHON_SCRIPT_PATH="../train_hypo.py" # Relative to scripts/ directory

echo "Starting Random Single MedMNIST-C training runs..."

for MODEL_ARCH in "${MODELS[@]}"; do
    for LOSS_TYPE in "hypo" "erm"; do
        for SEED in "${SEEDS[@]}"; do
            
            TRIAL_NAME_SUFFIX="rsmedc_${CORRUPTION_SOURCE}" # rsmedc = random single medmnist-c
            # Construct the trial name that train_hypo.py will use to form part of args.name
            TRIAL_FOR_PY_SCRIPT="${MODEL_ARCH}_${LOSS_TYPE}_${TRIAL_NAME_SUFFIX}_seed_${SEED}"

            echo "Running ${MODEL_ARCH} with ${LOSS_TYPE} (Random Single MedMNIST-C from ${CORRUPTION_SOURCE}), seed ${SEED}..."
            echo "Trial Name for logs: ${TRIAL_FOR_PY_SCRIPT}"

            CMD="python ${PYTHON_SCRIPT_PATH} \\
              --in-dataset ${DATASET} \\
              --wilds_root_dir ${WILDS_ROOT_DIR} \\
              --model ${MODEL_ARCH} \\
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
              --trial ${TRIAL_FOR_PY_SCRIPT} \\
              --seed ${SEED} \\
              --mode ${WANDB_MODE} \\
              --use_random_single_medmnistc \\
              --corruption_source_dataset ${CORRUPTION_SOURCE}"
            
            # Optional: Add --model_pretrained False if training from scratch is desired
            # CMD+=" \\ --model_pretrained False"

            echo "Command:"
            echo -e "$CMD\n"
            
            eval "$CMD"

            echo "${MODEL_ARCH} ${LOSS_TYPE} (Random Single MedMNIST-C) with seed ${SEED} finished."
            echo "----------------------------------------"
        done
    done
done

echo "All specified Random Single MedMNIST-C training runs completed."
