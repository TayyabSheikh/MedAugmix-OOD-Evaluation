import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import logging

# Import the augmented dataloader function
from dataloader.camelyon17_wilds_medmnistc import get_camelyon17_medmnistc_dataloaders

# Function to denormalize image tensor for visualization
def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Denormalizes an image tensor."""
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    tensor = tensor * std + mean  # Apply inverse transform
    tensor = torch.clamp(tensor, 0, 1) # Clamp values to [0, 1]
    return tensor

def visualize_augmented_samples(dataloader, num_samples=8, output_dir="visualizations", filename="medmnistc_dataloader_samples.png"):
    """Loads one batch, denormalizes, and visualizes samples."""
    if not dataloader:
        logging.error("Dataloader is None, cannot visualize.")
        return

    try:
        # Get one batch of data
        images, labels, metadata = next(iter(dataloader))
    except StopIteration:
        logging.error("Dataloader is empty.")
        return
    except Exception as e:
        logging.error(f"Error getting batch from dataloader: {e}")
        return

    # Ensure we don't try to plot more samples than available in the batch
    num_samples = min(num_samples, images.size(0))
    if num_samples == 0:
        logging.warning("No samples in the batch to visualize.")
        return

    logging.info(f"Visualizing {num_samples} augmented samples...")

    # Create figure
    cols = 4
    rows = (num_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten() # Flatten in case of single row/col

    for i in range(num_samples):
        img_tensor = images[i]
        label = labels[i].item()

        # Denormalize the image
        img_denorm = denormalize(img_tensor)

        # Convert to numpy array suitable for imshow (H, W, C)
        img_np = img_denorm.numpy().transpose((1, 2, 0))

        # Display the image
        ax = axes[i]
        ax.imshow(img_np)
        ax.set_title(f"Label: {label}")
        ax.axis('off')

    # Hide unused subplots
    for j in range(num_samples, len(axes)):
        axes[j].axis('off')

    plt.suptitle("Sample Augmented Images from Camelyon17 (MedMNIST-C Corruptions)")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap

    # Save the figure
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    try:
        plt.savefig(output_path)
        logging.info(f"Visualization saved to: {output_path}")
    except Exception as e:
        logging.error(f"Failed to save visualization: {e}")

    # Close the plot to free memory
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test and Visualize MedMNIST-C Augmented Dataloader for Camelyon17')
    parser.add_argument('--wilds_root_dir', default="./data", type=str, help='Root directory for WILDS datasets (relative to project root).')
    parser.add_argument('--batch_size', default=8, type=int, help='Batch size for visualization.')
    parser.add_argument('--num_workers', default=2, type=int, help='Number of dataloader workers.')
    parser.add_argument('--num_samples', default=8, type=int, help='Number of samples to visualize.')
    parser.add_argument('--output_dir', default="visualizations", type=str, help='Directory to save the visualization plot (relative to project root).')
    parser.add_argument('--filename', default="medmnistc_dataloader_samples.png", type=str, help='Filename for the saved plot.')
    parser.add_argument('--corruption_set', default="bloodmnist", type=str, help='MedMNIST corruption set to use (e.g., bloodmnist).')

    args = parser.parse_args()

    # Configure basic logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # Ensure matplotlib is available
    try:
        import matplotlib
        matplotlib.use('Agg') # Use Agg backend for non-interactive saving
        import matplotlib.pyplot as plt
    except ImportError:
        logging.error("Matplotlib is required for visualization. Please install it (`pip install matplotlib`).")
        exit()

    logging.info("Attempting to load augmented dataloader...")
    # Get only the training loader
    train_loader, _, _ = get_camelyon17_medmnistc_dataloaders(
        root_dir=args.wilds_root_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        corruption_dataset_name=args.corruption_set
    )

    if train_loader:
        visualize_augmented_samples(
            train_loader,
            num_samples=args.num_samples,
            output_dir=args.output_dir,
            filename=args.filename
        )
    else:
        logging.error("Failed to create dataloader. Cannot visualize.")
