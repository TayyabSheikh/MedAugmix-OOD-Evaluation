import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import wilds

def get_camelyon17_dataloaders(root_dir, batch_size, num_workers=4):
    """
    Prepares DataLoaders for the Camelyon17 dataset using WILDS.

    Args:
        root_dir (str): The directory where the dataset was downloaded.
        batch_size (int): The batch size for the DataLoaders.
        num_workers (int): Number of subprocesses to use for data loading.

    Returns:
        tuple: A tuple containing (train_loader, val_loader, test_loader).
               Returns (None, None, None) if the dataset cannot be loaded.
    """
    try:
        # Define standard transformations for Camelyon17
        # Input images are 96x96
        # Common practice is to resize and normalize using ImageNet stats
        image_size = 96 # Camelyon17 default patch size
        transform = transforms.Compose([
            transforms.ToTensor(), # Converts PIL image to tensor and scales to [0, 1]
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet stats
        ])

        # Get the full dataset object
        full_dataset = wilds.get_dataset(dataset='camelyon17', root_dir=root_dir, download=False) # Already downloaded

        # Get the training split
        train_data = full_dataset.get_subset(
            'train',
            transform=transform
        )
        train_loader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        print(f"Train loader created with {len(train_data)} samples.")

        # Get the validation split (used for OOD validation in WILDS)
        # Note: WILDS uses 'val' for ID validation and 'test' for OOD test.
        # The original HypO paper might use different splits depending on the benchmark.
        # Here we follow WILDS standard splits.
        val_data = full_dataset.get_subset(
            'val', # ID validation split
            transform=transform
        )
        val_loader = DataLoader(
            val_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        print(f"Validation (ID) loader created with {len(val_data)} samples.")


        # Get the test split (OOD test split in WILDS)
        test_data = full_dataset.get_subset(
            'test', # OOD test split
            transform=transform
        )
        test_loader = DataLoader(
            test_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        print(f"Test (OOD) loader created with {len(test_data)} samples.")

        return train_loader, val_loader, test_loader

    except Exception as e:
        print(f"Error creating Camelyon17 DataLoaders: {e}")
        return None, None, None

if __name__ == '__main__':
    # Example usage:
    print("Testing Camelyon17 DataLoader creation...")
    # Assuming the data is in ./data relative to the project root directory
    data_root = './data' # Corrected path relative to project root
    bs = 32
    train_dl, val_dl, test_dl = get_camelyon17_dataloaders(root_dir=data_root, batch_size=bs)

    if train_dl and val_dl and test_dl:
        print(f"\nSuccessfully created DataLoaders with batch size {bs}.")
        # Optional: Check a batch
        try:
            print("Checking one batch from train_loader...")
            for i, batch in enumerate(train_dl):
                # WILDS datasets return x, y, metadata
                x, y, metadata = batch
                print(f"Batch {i+1}:")
                print(f"  Input shape: {x.shape}") # Should be [batch_size, 3, 96, 96]
                print(f"  Target shape: {y.shape}") # Should be [batch_size]
                print(f"  Metadata keys: {metadata.keys()}")
                print(f"  Example metadata (hospital): {metadata['hospital'][0]}")
                break # Only check the first batch
            print("Batch check successful.")
        except Exception as e:
            print(f"Error checking batch: {e}")
    else:
        print("Failed to create DataLoaders.")
