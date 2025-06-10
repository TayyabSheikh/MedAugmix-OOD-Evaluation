import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import logging
from PIL import Image # Ensure PIL Image is imported if not already

# Import the new dataloader function
from dataloader.camelyon17_medmnistc_in_augmix import get_camelyon17_medmnistc_in_augmix_dataloaders

# Function to inverse normalize for visualization
def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Denormalizes a tensor image with mean and standard deviation."""
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    tensor = tensor * std + mean # Apply inverse transform
    tensor = torch.clamp(tensor, 0, 1) # Clamp values to [0, 1] range
    return tensor

def visualize_samples(dataloader, num_samples=8, filename="medmnistc_augmix_samples.png", output_dir="../visualizations"):
    """Loads a batch, denormalizes, and saves a grid of images."""
    if dataloader is None:
        print("Dataloader is None, cannot visualize.")
        return

    # Get one batch of data
    try:
        images, labels, _ = next(iter(dataloader))
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
    denormalized_samples = [denormalize(img) for img in samples]

    # Plotting
    fig, axes = plt.subplots(1, num_samples, figsize=(num_samples * 2, 3))
    if num_samples == 1: # Handle case where only one sample is plotted
        axes = [axes]
    fig.suptitle('Sample Training Images (MedMNIST-C in AugMix)', fontsize=16)

    for i in range(num_samples):
        img_np = denormalized_samples[i].numpy().transpose((1, 2, 0)) # Convert to HWC format for plotting
        axes[i].imshow(img_np)
        axes[i].set_title(f'Label: {labels[i].item()}')
        axes[i].axis('off')

    # Save the figure
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath)
    print(f"Visualization saved to {filepath}")
    plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize MedMNIST-C + AugMix samples from Camelyon17 dataloader')
    parser.add_argument('--wilds_root_dir', default="../data", type=str, help='Root directory for WILDS datasets (relative to script location).')
    parser.add_argument('--batch_size', default=8, type=int, help='Batch size for loading data (should be >= num_samples)')
    parser.add_argument('--num_samples', default=8, type=int, help='Number of samples to visualize')
    parser.add_argument('--corruption_dataset', default="bloodmnist", type=str, help='MedMNIST-C corruption set to use for AugMix operations')
    parser.add_argument('--output_dir', default="../visualizations", type=str, help='Directory to save the visualization')
    parser.add_argument('--output_filename', default="medmnistc_in_augmix_samples.png", type=str, help='Filename for the saved visualization')
    parser.add_argument('--use_med_augmix', action='store_true', default=True, help='Enable MedMNIST-C operations within AugMix (should be true for this script)')
    parser.add_argument('--augmix_severity', type=int, default=3, help='Severity for AugMix operations')
    parser.add_argument('--augmix_mixture_width', type=int, default=3, help='Mixture width for AugMix')


    args = parser.parse_args()

    # Configure basic logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    print("Creating dataloader with MedMNIST-C operations in AugMix enabled...")
    # Get only the training loader with MedAugMix enabled
    train_loader, _, _ = get_camelyon17_medmnistc_in_augmix_dataloaders(
        root_dir=args.wilds_root_dir,
        batch_size=args.batch_size,
        num_workers=1, # Only need 1 worker for visualization
        corruption_dataset_name=args.corruption_dataset,
        use_med_augmix=args.use_med_augmix,
        augmix_severity=args.augmix_severity,
        augmix_mixture_width=args.augmix_mixture_width
    )

    # Visualize the samples
    visualize_samples(
        dataloader=train_loader,
        num_samples=args.num_samples,
        filename=args.output_filename,
        output_dir=args.output_dir
    )

    print("Visualization script finished.")
