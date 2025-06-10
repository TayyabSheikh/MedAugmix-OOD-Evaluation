import numpy as np
import os
from collections import Counter

# Assuming fitzpatrick17k_loader.py is in the same directory or accessible in PYTHONPATH
# For direct execution, adjust path if necessary or ensure dataloader module is found
try:
    from dataloader.fitzpatrick17k_loader import Fitzpatrick17k
except ImportError:
    # Fallback for direct execution from hypo_impl folder
    from fitzpatrick17k_loader import Fitzpatrick17k


def analyze_fst_distribution(dataset_root_dir):
    """
    Analyzes and prints the distribution of images across Fitzpatrick Scale groups.

    Args:
        dataset_root_dir (str): The root directory of the Fitzpatrick17k dataset
                                (e.g., path to 'finalfitz17k').
    """
    print(f"Analyzing Fitzpatrick17k dataset at: {dataset_root_dir}")

    # Instantiate the dataset to load all metadata
    # split='all' ensures all data is loaded by _load_images before any splitting logic
    # label_partition and target_domain are not critical for this specific analysis of FST groups
    # as long as _load_images populates self.fiz_scales correctly.
    try:
        dataset = Fitzpatrick17k(
            root=dataset_root_dir,
            split='all',  # Load all data for analysis
            label_partition=3, # Arbitrary valid value, not used for FST counting
            target_domain=None # Not splitting by target for this analysis
        )
    except Exception as e:
        print(f"Error initializing Fitzpatrick17k dataset: {e}")
        print("Please ensure the dataset path is correct and fitzpatrick17k.csv exists.")
        return

    if not hasattr(dataset, 'fiz_scales') or not dataset.fiz_scales.size > 0:
        print("No Fitzpatrick scale data loaded. Check dataset integrity and _load_images method.")
        return

    fst_groups = dataset.fiz_scales
    counts = Counter(fst_groups)

    print("\nImage Counts per Fitzpatrick Scale Group:")
    print("---------------------------------------")
    total_images = 0
    for group_val, count in sorted(counts.items()):
        group_name = ""
        if group_val == 12:
            group_name = "FST Group 1 & 2 (FSG1)"
        elif group_val == 34:
            group_name = "FST Group 3 & 4 (FSG2)"
        elif group_val == 56:
            group_name = "FST Group 5 & 6 (FSG3)"
        else:
            group_name = f"Unknown Group ({group_val})" # Should not happen with current loader
        
        print(f"{group_name}: {count} images")
        total_images += count
    
    print("---------------------------------------")
    print(f"Total images with valid FST (1-6): {total_images}")

    if total_images > 0:
        print("\nPercentages:")
        for group_val, count in sorted(counts.items()):
            group_name = ""
            if group_val == 12: group_name = "FSG1"
            elif group_val == 34: group_name = "FSG2"
            elif group_val == 56: group_name = "FSG3"
            percentage = (count / total_images) * 100
            print(f"{group_name}: {percentage:.2f}%")
    print("---------------------------------------")


if __name__ == "__main__":
    # IMPORTANT: Adjust this path to the root of your 'finalfitz17k' dataset directory
    # Example: dataset_path = "/path/to/your/data/finalfitz17k"
    # Using relative path assuming the script is run from hypo_impl directory
    # and 'data' is a sibling of 'hypo_impl'
    
    # Construct path relative to the script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_dataset_path = os.path.join(script_dir, '..', 'data', 'finalfitz17k') # Goes up one level from hypo_impl/ to Thesis/, then data/finalfitz17k

    dataset_path_to_analyze = default_dataset_path
    
    # You can override with an absolute path if needed:
    # dataset_path_to_analyze = "/home/tsheikh/Thesis/data/finalfitz17k" 

    if not os.path.isdir(dataset_path_to_analyze):
        print(f"Error: Dataset directory not found at '{dataset_path_to_analyze}'.")
        print("Please update the 'dataset_path_to_analyze' variable in this script.")
    else:
        analyze_fst_distribution(dataset_path_to_analyze)
