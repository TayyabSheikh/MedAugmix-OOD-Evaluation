#!/bin/bash

# Script to run specific missing seeds for experimental configurations.
# - Configurations reported as complete (5+ seeds) are skipped.
# - For configurations reported with 2 seeds, seeds 2 and 3 will be actively run.
# - For other incomplete configurations, commands for seeds 0-4 are generated commented out for manual review.

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

# --- Seeds to run for incomplete configurations ---
SEEDS_TO_RUN_FOR_INCOMPLETE=(0 1 2)

# --- Unique Base Trial Configurations (Extracted from report) ---
# Format: MODEL LOSS_TYPE AUGMENTATION_STRATEGY [AUG_SEV] [AUG_WIDTH]
# This array will be populated based on the report.
# For now, I'll list them based on the provided report.
# The script will parse these to set appropriate flags.

declare -A CONFIG_DETAILS

# Configurations marked with #_COMPLETE_# are considered complete based on the report and will be skipped.
CONFIG_DETAILS["densenet121_erm_medaugmix_sev5_w1"]="#_COMPLETE_#densenet121 erm medmnist_c_augmix 5 1"
CONFIG_DETAILS["densenet121_erm_medaugmix_sev3_w5"]="#_COMPLETE_#densenet121 erm medmnist_c_augmix 3 5"
CONFIG_DETAILS["densenet121_hypo_medaugmix_sev5_w1"]="#_COMPLETE_#densenet121 hypo medmnist_c_augmix 5 1"
CONFIG_DETAILS["densenet121_hypo_medaugmix_sev3_w5"]="#_COMPLETE_#densenet121 hypo medmnist_c_augmix 3 5"
CONFIG_DETAILS["resnet50_erm_medaugmix_sev5_w1"]="#_COMPLETE_#resnet50 erm medmnist_c_augmix 5 1" 
CONFIG_DETAILS["densenet121_erm_plain_augmix"]="#_COMPLETE_#densenet121 erm plain_augmix"
CONFIG_DETAILS["densenet121_hypo_medmnist_c"]="densenet121 hypo medmnist_c" # Reported with 2 seeds
CONFIG_DETAILS["densenet121_hypo_plain_augmix"]="#_COMPLETE_#densenet121 hypo plain_augmix"
CONFIG_DETAILS["densenet121_erm_medmnist_c"]="densenet121 erm medmnist_c" # Reported with 2 seeds (this is the active one due to duplicate key)
CONFIG_DETAILS["densenet121_hypo_baseline"]="densenet121 hypo baseline" # Reported with 3 seeds
CONFIG_DETAILS["densenet121_erm_baseline"]="densenet121 erm baseline" # Reported with 3 seeds
CONFIG_DETAILS["resnet50_erm_medaugmix_sev3_w5"]="#_COMPLETE_#resnet50 erm medmnist_c_augmix 3 5"
# resnet50_hypo_medaugmix_sev5_w1 is listed below and is incomplete
CONFIG_DETAILS["resnet50_hypo_medaugmix_sev5_w1"]="resnet50 hypo medmnist_c_augmix 5 1" # Reported with 3 seeds
CONFIG_DETAILS["resnet50_erm_baseline"]="resnet50 erm baseline" # Reported with 3 seeds
CONFIG_DETAILS["resnet50_erm_medmnist_c"]="resnet50 erm medmnist_c" # Reported with 2 seeds
CONFIG_DETAILS["resnet50_hypo_medaugmix_sev3_w5"]="#_COMPLETE_#resnet50 hypo medmnist_c_augmix 3 5"
CONFIG_DETAILS["resnet50_hypo_baseline"]="resnet50 hypo baseline" # Reported with 3 seeds
CONFIG_DETAILS["resnet50_hypo_medmnist_c"]="resnet50 hypo medmnist_c" # Reported with 2 seeds


echo "Processing configurations..."
echo "- Seeds 2 and 3 will be actively run for configurations previously reported with 2 seeds."
echo "- Other incomplete configurations will have commands for seeds ${SEEDS_TO_RUN_FOR_INCOMPLETE[*]} generated (commented out)."
echo "- Review and manually uncomment other specific seed commands if needed."

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

for base_trial_name in "${!CONFIG_DETAILS[@]}"; do
    CONFIG_STRING="${CONFIG_DETAILS[$base_trial_name]}"

    echo "" # Add a blank line for better readability between configurations
    echo "# --- Processing Configuration: ${base_trial_name} ---"

    if [[ "$CONFIG_STRING" == "#_COMPLETE_#"* ]]; then
        echo "# Reported as complete. Skipping command generation for this configuration."
        echo "# ----------------------------------------"
        continue
    else
        echo "# Reported as incomplete or not found with 5 seeds. Generating commands."
        echo "# Please review and comment out specific seeds below if they are already complete."
        # Remove the prefix for actual parsing if we decide to run it
        ACTUAL_CONFIG_PARAMS="${CONFIG_STRING#\#_INCOMPLETE_#}" # In case we used a different marker, ensure it's clean
        IFS=' ' read -r MODEL LOSS_TYPE AUG_STRATEGY AUG_SEV AUG_WIDTH <<< "$ACTUAL_CONFIG_PARAMS"
    fi

    PYTHON_SCRIPT=""
    AUG_FLAGS=""

    case "$AUG_STRATEGY" in
        baseline)
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
            PYTHON_SCRIPT="../train_plain_augmix.py" # This script applies plain AugMix by default
            ;;
        medmnist_c_augmix)
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

    for SEED in "${SEEDS_TO_RUN_FOR_INCOMPLETE[@]}"; do
        TRIAL="${base_trial_name}_seed_${SEED}" # Construct full trial name
        
        echo "# Preparing to run: Model=${MODEL}, Loss=${LOSS_TYPE}, Aug=${AUG_STRATEGY}, Sev=${AUG_SEV:-N/A}, Width=${AUG_WIDTH:-N/A}, Seed=${SEED}"
        echo "# Trial Name: ${TRIAL}"
        echo "# Python Script: ${PYTHON_SCRIPT}"

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
        
        echo "# Command for ${TRIAL}:"
        # Indent the command for clarity when printed
        
        IS_TWO_SEED_CONFIG=false
        if [[ "$base_trial_name" == "densenet121_hypo_medmnist_c" || \
              "$base_trial_name" == "densenet121_erm_medmnist_c" || \
              "$base_trial_name" == "resnet50_erm_medmnist_c" || \
              "$base_trial_name" == "resnet50_hypo_medmnist_c" ]]; then
            IS_TWO_SEED_CONFIG=true
        fi

        if [[ "$IS_TWO_SEED_CONFIG" == true && ( "$SEED" == "2" || "$SEED" == "3" ) ]]; then
            echo "# This command WILL BE EXECUTED for ${base_trial_name} (seed $SEED)."
            echo -e "${CMD//\\$'\n'/\n}" # Print command without initial '#'
            echo "eval \"\$CMD\"" # Eval command is active
            echo ""
            eval "$CMD" # Execute the command
            echo "# Finished active run for: ${TRIAL}"
        else
            echo "# This command is COMMENTED OUT for ${base_trial_name} (seed $SEED)."
            COMMENTED_CMD=$(echo "$CMD" | sed 's/^/# /')
            echo -e "${COMMENTED_CMD//\\$'\n'/\n# }" # Ensure backslashes for newlines are also commented
            echo "# eval \"\$CMD\"" # Show the eval command commented
            echo ""
        fi
        
        # echo "# Finished processing for: ${TRIAL}" # Generic message
        echo "# ---------------------------------------- (End of commands for SEED ${SEED})"
    done
    echo "# ---------------------------------------- (End of commands for ${base_trial_name})"
done

echo ""
echo "Script processing complete."
echo "Script processing complete."
echo "Review the output above."
echo "- For configurations reported with 2 seeds, seeds 2 and 3 were actively run."
echo "- For other configurations not marked as '#_COMPLETE_#', commands for seeds ${SEEDS_TO_RUN_FOR_INCOMPLETE[*]} were generated but are commented out."
echo "You may need to manually edit and uncomment 'eval \"\$CMD\"' lines for other specific seeds you wish to run."
