import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import wilds
import logging

# Import necessary components from medmnistc
try:
    from medmnistc.augmentation import AugMedMNISTC
    from medmnistc.corruptions.registry import CORRUPTIONS_DS
    medmnistc_available = True
except ImportError:
    logging.warning("medmnistc library not found. AugMedMNISTC augmentation will not be available.")
    medmnistc_available = False
    # Define dummy classes/variables if medmnistc is not installed to avoid errors later
    class AugMedMNISTC:
        def __init__(self, *args, **kwargs):
            logging.error("Attempted to use AugMedMNISTC, but medmnistc is not installed.")
        def __call__(self, img):
            # Return image unchanged if library is missing
            return img
    CORRUPTIONS_DS = {}


def get_camelyon17_medmnistc_dataloaders(root_dir, batch_size, num_workers=4, corruption_dataset_name="bloodmnist"):
    """
    Prepares DataLoaders for the Camelyon17 dataset using WILDS,
    applying MedMNIST-C corruptions (specified by corruption_dataset_name)
    to the training set.

    Args:
        root_dir (str): The directory where the Camelyon17 dataset was downloaded.
        batch_size (int): The batch size for the DataLoaders.
        num_workers (int): Number of subprocesses to use for data loading.
        corruption_dataset_name (str): The MedMNIST dataset name whose corruption set
                                       should be used (e.g., "bloodmnist").

    Returns:
        tuple: A tuple containing (train_loader, val_loader, test_loader).
               Returns (None, None, None) if the dataset cannot be loaded or
               if medmnistc is required but not installed.
    """
    if not medmnistc_available:
        logging.error("Cannot create augmented dataloader because medmnistc library is not installed.")
        return None, None, None

    try:
        # Define standard transformations (applied AFTER potential augmentation)
        # Input images are 96x96
        # Common practice is to resize and normalize using ImageNet stats
        base_transforms = transforms.Compose([
            transforms.ToTensor(), # Converts PIL image to tensor and scales to [0, 1]
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet stats
        ])

        # Get the corruption set based on the specified dataset name
        if corruption_dataset_name not in CORRUPTIONS_DS:
            logging.error(f"Corruption set for '{corruption_dataset_name}' not found in medmnistc.CORRUPTIONS_DS.")
            logging.info(f"Available sets: {list(CORRUPTIONS_DS.keys())}")
            return None, None, None
        train_corruptions = CORRUPTIONS_DS[corruption_dataset_name]
        logging.info(f"Using MedMNIST-C corruptions defined for: {corruption_dataset_name}")

        # Define augmented transformations for the training set
        # Apply MedMNIST-C augmentation FIRST (expects PIL image), then base transforms
        augmented_transforms = transforms.Compose([
            AugMedMNISTC(train_corruptions), # Apply the targeted corruptions
            *base_transforms.transforms # Include ToTensor and Normalize
        ])

        # Get the full dataset object
        logging.info(f"Loading Camelyon17 dataset from: {root_dir}")
        full_dataset = wilds.get_dataset(dataset='camelyon17', root_dir=root_dir, download=False) # Assume already downloaded

        # Get the training split with augmented transforms
        train_data = full_dataset.get_subset(
            'train',
            transform=augmented_transforms # Apply augmentations here
        )
        train_loader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        logging.info(f"Train loader created with {len(train_data)} samples (MedMNIST-C augmented).")

        # Get the validation split with standard transforms
        val_data = full_dataset.get_subset(
            'val', # ID validation split
            transform=base_transforms # Standard transforms for validation
        )
        val_loader = DataLoader(
            val_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        logging.info(f"Validation (ID) loader created with {len(val_data)} samples (standard transforms).")

        # Get the test split with standard transforms
        test_data = full_dataset.get_subset(
            'test', # OOD test split
            transform=base_transforms # Standard transforms for testing
        )
        test_loader = DataLoader(
            test_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        logging.info(f"Test (OOD) loader created with {len(test_data)} samples (standard transforms).")

        return train_loader, val_loader, test_loader

    except Exception as e:
        logging.error(f"Error creating Camelyon17 MedMNIST-C DataLoaders: {e}", exc_info=True)
        return None, None, None

if __name__ == '__main__':
    # Example usage:
    print("Testing Camelyon17 MedMNIST-C DataLoader creation...")
    # Configure logging for the example
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # Assuming the data is in ./data relative to the project root directory
    # Adjust this path as needed
    data_root = '../data' # Corrected path relative to hypo_impl/dataloader/
    bs = 32
    num_w = 2

    # Test with bloodmnist corruptions
    train_dl, val_dl, test_dl = get_camelyon17_medmnistc_dataloaders(
        root_dir=data_root,
        batch_size=bs,
        num_workers=num_w,
        corruption_dataset_name="bloodmnist"
    )

    if train_dl and val_dl and test_dl:
        print(f"\nSuccessfully created DataLoaders with batch size {bs}.")
        # Optional: Check a batch from the augmented train_loader
        try:
            print("Checking one batch from augmented train_loader...")
            for i, batch in enumerate(train_dl):
                # WILDS datasets return x, y, metadata
                x, y, metadata = batch
                print(f"Batch {i+1}:")
                print(f"  Input shape: {x.shape}") # Should be [batch_size, 3, 96, 96]
                print(f"  Target shape: {y.shape}") # Should be [batch_size]
                print(f"  Metadata keys: {metadata.keys()}")
                print(f"  Example metadata (hospital): {metadata['hospital'][0]}")
                # You might want to visualize x[0] here to confirm augmentation
                break # Only check the first batch
            print("Augmented batch check successful.")
        except Exception as e:
            print(f"Error checking augmented batch: {e}")
    else:
        print("Failed to create MedMNIST-C DataLoaders.")
