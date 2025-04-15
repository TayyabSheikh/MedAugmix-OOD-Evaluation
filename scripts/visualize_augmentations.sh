#!/bin/bash

# Script to run the MedMNIST-C augmentation visualization for Camelyon17.

# --- Configuration ---
WILDS_ROOT_DIR="../../data" # Relative path from scripts/ to data/
OUTPUT_DIR="../../visualizations" # Relative path from scripts/ to visualizations/
FILENAME="medmnistc_camelyon17_samples.png"
NUM_SAMPLES=32 # Number of samples to show in the plot
BATCH_SIZE=32  # Batch size for the dataloader (must be >= NUM_SAMPLES)
NUM_WORKERS=2
CORRUPTION_SET="bloodmnist" # Corruption set to visualize

# --- Activate Conda Environment (Optional - uncomment if needed) ---
# source $(conda info --base)/etc/profile.d/conda.sh
# conda activate your_env_name

# --- Construct and Run Command ---
# Assumes the script is run from the hypo_impl/scripts directory

echo "Running MedMNIST-C augmentation visualization script..."
echo "Output directory: ${OUTPUT_DIR}"
echo "Filename: ${FILENAME}"
echo "Command:"

# Print the command before executing
echo "python ../test_visualize_medmnistc_dataloader.py \\"
echo "  --wilds_root_dir ${WILDS_ROOT_DIR} \\"
echo "  --output_dir ${OUTPUT_DIR} \\"
echo "  --filename ${FILENAME} \\"
echo "  --num_samples ${NUM_SAMPLES} \\"
echo "  --batch_size ${BATCH_SIZE} \\"
echo "  --num_workers ${NUM_WORKERS} \\"
echo "  --corruption_set ${CORRUPTION_SET}"

# Execute the command directly
python ../test_visualize_medmnistc_dataloader.py \
  --wilds_root_dir ${WILDS_ROOT_DIR} \
  --output_dir ${OUTPUT_DIR} \
  --filename ${FILENAME} \
  --num_samples ${NUM_SAMPLES} \
  --batch_size ${BATCH_SIZE} \
  --num_workers ${NUM_WORKERS} \
  --corruption_set ${CORRUPTION_SET}

echo "Visualization script finished."
echo "Output saved to ${OUTPUT_DIR}/${FILENAME}"
