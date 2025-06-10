import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import logging
from PIL import Image

# Import the FitzPatrick17k dataloader function
from dataloader.fitzpatrick17k_dataloaders_utils import get_fitzpatrick17k_dataloaders

# Function to inverse normalize for visualization
def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Denormalizes a tensor image with mean and standard deviation."""
    mean_tensor = torch.tensor(mean).view(3, 1, 1)
    std_tensor = torch.tensor(std).view(3, 1, 1)
    if tensor.is_cuda:
        mean_tensor = mean_tensor.to(tensor.device)
        std_tensor = std_tensor.to(tensor.device)
    tensor = tensor * std_tensor + mean_tensor # Apply inverse transform
    tensor = torch.clamp(tensor, 0, 1) # Clamp values to [0, 1] range
    return tensor

def visualize_samples(dataloader, num_samples=8, filename="fitzpatrick_augmix_samples.png", output_dir="../visualizations", dataset_name="Fitzpatrick17k"):
    """Loads a batch, denormalizes, and saves a grid of images."""
    if dataloader is None:
        print("Dataloader is None, cannot visualize.")
        return

    # Get one batch of data
    try:
        # Fitzpatrick17k loader returns: img, label, fs_group (domain_info)
        images, labels, domain_info = next(iter(dataloader))
    except StopIteration:
        print("Could not get a batch from the dataloader.")
        return
    except Exception as e:
        print(f"Error getting batch from dataloader: {e}")
        return

    if images.shape[0] < num_samples:
        print(f"Warning: Batch size ({images.shape[0]}) is smaller than requested num_samples ({num_samples}). Visualizing available samples.")
        num_samples = images.shape[0]

    # Select samples and denormalize
    samples = images[:num_samples]
    denormalized_samples = [denormalize(img.cpu()) for img in samples] # Ensure tensor is on CPU for numpy conversion

    # Plotting
    fig, axes = plt.subplots(1, num_samples, figsize=(num_samples * 2.5, 3.5)) # Adjusted figsize
    if num_samples == 1: # Handle case where only one sample is plotted
        axes = [axes]
    fig.suptitle(f'Sample Training Images ({dataset_name} with AugMix)', fontsize=16)

    for i in range(num_samples):
        img_np = denormalized_samples[i].numpy().transpose((1, 2, 0)) # Convert to HWC format for plotting
        axes[i].imshow(img_np)
        title_parts = [f'Label: {labels[i].item()}']
        if domain_info is not None and i < len(domain_info):
            title_parts.append(f'FST: {domain_info[i].item()}')
        axes[i].set_title("\n".join(title_parts))
        axes[i].axis('off')

    # Save the figure
    try:
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
        print(f"Attempting to save visualization to: {filepath}")
        plt.savefig(filepath)
        print(f"Visualization saved to {filepath}")
    except Exception as e:
        print(f"Error saving visualization to {filepath}: {e}")
    finally:
        plt.close(fig)

if __name__ == "__main__":
    # Determine a robust default output directory relative to the script's location
    # Script is in hypo_impl/, so ../visualizations points to Thesis/visualizations/
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_viz_dir = os.path.join(script_dir, "..", "visualizations")

    parser = argparse.ArgumentParser(description='Visualize AugMix samples from Fitzpatrick17k dataloader')
    parser.add_argument('--wilds_root_dir', default=os.path.join(script_dir, "..", "data", "finalfitz17k"), type=str, help='Root directory for Fitzpatrick17k dataset.')
    parser.add_argument('--batch_size', default=8, type=int, help='Batch size for loading data (should be >= num_samples)')
    parser.add_argument('--num_samples', default=8, type=int, help='Number of samples to visualize')
    parser.add_argument('--output_dir', default=default_viz_dir, type=str, help='Directory to save the visualization')
    parser.add_argument('--output_filename', default="fitzpatrick17k_augmix_samples.png", type=str, help='Filename for the saved visualization')
    
    # Fitzpatrick17k specific arguments
    parser.add_argument('--label_partition', type=int, default=3, choices=[3, 9, 114], help='Label partition for Fitzpatrick17k (3, 9, or 114 classes)')
    parser.add_argument('--target_domain_ood_test', type=str, default='56', help='Target domain FST group for OOD test (e.g., 12, 34, 56), not directly used for train viz but needed by dataloader.')
    parser.add_argument('--img_size', type=int, default=224, help='Image size for transforms.')

    # AugMix arguments (passed to get_fitzpatrick17k_dataloaders)
    parser.add_argument('--use_med_augmix', action='store_true', default=True, help='Enable MedMNIST-C operations within AugMix (should be true for this script to show AugMix effects)')
    parser.add_argument('--augmix_corruption_dataset', default="dermamnist", type=str, help='MedMNIST-C corruption set to use for AugMix operations')
    parser.add_argument('--augmix_severity', type=int, default=3, help='Severity for AugMix operations')
    parser.add_argument('--augmix_mixture_width', type=int, default=3, help='Mixture width for AugMix')
    # --augment flag from train_hypo_fitzpatrick.py is not directly used here, 
    # behavior is controlled by use_med_augmix for the dataloader.
    # We can add it if we want to visualize non-MedAugMix augmentations too.
    # For now, focusing on visualizing MedAugMix.

    args = parser.parse_args()

    # Configure basic logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    print("Creating Fitzpatrick17k dataloader with MedAugMix enabled...")
    
    train_loader, _, _ = get_fitzpatrick17k_dataloaders(
        root_dir=args.wilds_root_dir,
        batch_size=args.batch_size,
        num_workers=1, # Only need 1 worker for visualization
        label_partition=args.label_partition,
        target_domain_ood_test=args.target_domain_ood_test,
        augment_train=True, # Set to True so dataloader considers augmentation pipeline. If use_med_augmix is True, it takes precedence.
        img_size=args.img_size,
        use_med_augmix=args.use_med_augmix,
        augmix_corruption_dataset=args.augmix_corruption_dataset,
        augmix_severity=args.augmix_severity,
        augmix_mixture_width=args.augmix_mixture_width
    )

    # Visualize the samples
    visualize_samples(
        dataloader=train_loader,
        num_samples=args.num_samples,
        filename=args.output_filename,
        output_dir=args.output_dir,
        dataset_name=f"Fitzpatrick17k (LP{args.label_partition})"
    )

    print("Visualization script finished.")
