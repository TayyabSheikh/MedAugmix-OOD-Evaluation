import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import wilds
import logging
from PIL import Image
# Remove plt import as we remove visualization from this file's test block
# import matplotlib.pyplot as plt 

# Import necessary components from medmnistc
try:
    from medmnistc.corruptions.registry import CORRUPTIONS_DS
    medmnistc_available = True
    # Import our custom AugMix implementation
    from .custom_augmentations import MedMNISTCAugMix
except ImportError as e:
    # Handle import errors for both medmnistc and custom_augmentations
    logging.warning(f"ImportError: {e}. MedMNIST-C corruptions or MedMNISTCAugMix might not be available.")
    logging.warning("medmnistc library not found. MedMNIST-C corruptions for AugMix will not be available.")
    medmnistc_available = False
    CORRUPTIONS_DS = {}

def get_camelyon17_medmnistc_in_augmix_dataloaders(root_dir, batch_size, num_workers=4, corruption_dataset_name="bloodmnist", use_med_augmix=False, augmix_severity=3, augmix_mixture_width=3):
    """
    Prepares DataLoaders for Camelyon17, optionally applying AugMix where
    the AugMix operations are sourced from MedMNIST-C corruptions.

    Args:
        root_dir (str): Directory for Camelyon17 dataset.
        batch_size (int): Batch size.
        num_workers (int): Number of data loading workers.
        corruption_dataset_name (str): MedMNIST dataset name for corruption set (e.g., "bloodmnist").
        use_med_augmix (bool): If True, apply AugMix with MedMNIST-C operations.
        augmix_severity (int): Severity parameter for AugMix.
        augmix_mixture_width (int): Mixture width parameter for AugMix.


    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    if use_med_augmix and not medmnistc_available:
        logging.error("Cannot use MedAugMix because medmnistc library is not installed.")
        return None, None, None

    try:
        base_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        train_transform_list = []

        if use_med_augmix:
            if corruption_dataset_name not in CORRUPTIONS_DS:
                logging.error(f"Corruption set for '{corruption_dataset_name}' not found in medmnistc.CORRUPTIONS_DS.")
                return None, None, None
            
            medmnistc_ops = CORRUPTIONS_DS[corruption_dataset_name]
            logging.info(f"Applying AugMix with MedMNIST-C operations from '{corruption_dataset_name}' set.")

            # Instantiate our custom MedMNISTCAugMix
            # It handles the ops internally based on corruption_dataset_name
            try:
                 med_augmix_transform = MedMNISTCAugMix(
                     corruption_dataset_name=corruption_dataset_name,
                     severity=augmix_severity,
                     mixture_width=augmix_mixture_width
                     # Add mixture_depth and alpha if needed/supported by MedMNISTCAugMix
                 )
                 train_transform_list.append(med_augmix_transform)
            except (ImportError, ValueError) as e:
                 logging.error(f"Could not initialize MedMNISTCAugMix: {e}")
                 return None, None, None # Fail dataloader creation if custom AugMix fails
        else:
            # Fallback or standard augmentation if not using MedAugMix
            # For this specific dataloader, if not use_med_augmix, we might apply no extra aug or standard ones.
            # Let's assume no other augmentation if use_med_augmix is False for this specialized loader.
            logging.info("MedAugMix not applied. Applying only base transforms for training.")


        # Add base transforms (ToTensor, Normalize) at the end
        train_transform_list.extend(base_transforms.transforms)
        augmented_transforms = transforms.Compose(train_transform_list)

        logging.info(f"Loading Camelyon17 dataset from: {root_dir}")
        full_dataset = wilds.get_dataset(dataset='camelyon17', root_dir=root_dir, download=False)

        train_data = full_dataset.get_subset('train', transform=augmented_transforms)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        logging.info(f"Train loader created with {len(train_data)} samples.")

        val_data = full_dataset.get_subset('val', transform=base_transforms)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        logging.info(f"Validation (ID) loader created with {len(val_data)} samples.")

        test_data = full_dataset.get_subset('test', transform=base_transforms)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        logging.info(f"Test (OOD) loader created with {len(test_data)} samples.")

        return train_loader, val_loader, test_loader

    except Exception as e:
        logging.error(f"Error creating Camelyon17 MedMNIST-C-in-AugMix DataLoaders: {e}", exc_info=True)
        return None, None, None

if __name__ == '__main__':
    # Simplified test block: Just try creating the dataloader
    print("Testing Camelyon17 MedMNIST-C-in-AugMix DataLoader creation (using custom MedMNISTCAugMix)...")
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # Define necessary paths and parameters for testing
    # Assuming script is run from project root (e.g., /home/tsheikh/Thesis)
    # Adjust relative paths accordingly if run from hypo_impl/dataloader/
    data_root = './data' # Path relative to project root
    bs = 8
    num_w = 1

    # Test with MedAugMix enabled (using our custom class)
    train_dl, _, _ = get_camelyon17_medmnistc_in_augmix_dataloaders(
        root_dir=data_root,
        batch_size=bs,
        num_workers=num_w,
        corruption_dataset_name="bloodmnist",
        use_med_augmix=True
    )

    if train_dl:
        print(f"\nSuccessfully created DataLoader with MedAugMix (batch size {bs}).")
    if train_dl:
        print(f"\nSuccessfully created DataLoader with MedAugMix (batch size {bs}).")
        try:
            print("Checking one batch can be loaded...")
            for i, batch in enumerate(train_dl):
                x, y, metadata = batch
                print(f"Batch {i+1}: Input shape: {x.shape}, Target shape: {y.shape}")
                # Basic check passed if we get here
                break # Only check the first batch
            print("Batch loading check successful.")
        except Exception as e:
            print(f"Error loading batch: {e}")
    else:
        print("Failed to create MedAugMix DataLoader.")
