import torch
import torchvision.transforms as transforms
from .fitzpatrick17k_loader import Fitzpatrick17k
from .custom_augmentations import MedMNISTCAugMix, MedMNISTCorruptionTransform, medmnistc_available, get_medmnistc_corruption_names_by_collection
import logging
import random

def get_fitzpatrick17k_dataloaders(
    root_dir, batch_size, num_workers, 
    label_partition, target_domain_ood_test, 
    augment_train=False, img_size=224,
    use_med_augmix=False, 
    augmix_corruption_dataset="dermamnist",
    augmix_severity=3, 
    augmix_mixture_width=3,
    use_torchvision_augmix=False, 
    tv_augmix_severity=3,        
    tv_augmix_mixture_width=3,  
    tv_augmix_alpha=1.0,
    use_plain_medmnistc=False, # New arg for plain MedMNISTC
    # plain_medmnistc_corruption_name is now implicitly 'random_from_dermamnist'
    # plain_medmnistc_severity is now implicitly 'random_1_to_5'
    # We can add specific args later if needed, for now, randomizing if use_plain_medmnistc is True
    plain_medmnistc_collection_source="dermamnist" # Source for random corruptions
    ):
    """
    Prepares and returns DataLoaders for the Fitzpatrick17k dataset.
    """
    normalize_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
    
    train_transform_list = []

    if use_plain_medmnistc:
        logging.info(f"Applying Plain MedMNISTC: Random corruption from '{plain_medmnistc_collection_source}' with random severity (1-5).")
        if medmnistc_available:
            # This will be a transform that, when called, picks a random corruption and severity
            # We need a new wrapper or to modify MedMNISTCorruptionTransform to support this.
            # For now, let's assume MedMNISTCorruptionTransform can handle name="random" and severity="random"
            # Or, more practically, create a new transform class here.

            class RandomPlainMedMNISTCTransform:
                def __init__(self, collection_source="dermamnist", img_size_for_resize=224):
                    self.collection_source = collection_source
                    self.corruption_names = get_medmnistc_corruption_names_by_collection(collection_source)
                    if not self.corruption_names:
                        logging.error(f"No corruption names found for collection {collection_source}. Plain MedMNISTC cannot be applied.")
                        # Fallback to basic resize/crop if corruptions can't be loaded
                        self.transform_chain = transforms.Compose([
                            transforms.Resize(img_size_for_resize),
                            transforms.CenterCrop(img_size_for_resize)
                        ])
                    else:
                        logging.info(f"Plain MedMNISTC will randomly select from: {self.corruption_names}")
                    self.img_size_for_resize = img_size_for_resize


                def __call__(self, img):
                    if not self.corruption_names: # Fallback if list is empty
                        return self.transform_chain(img)

                    chosen_corruption_name = random.choice(self.corruption_names)
                    chosen_severity = random.randint(1, 5) # MedMNIST-C severities are 1-5
                    
                    # Use MedMNISTCorruptionTransform to apply the single chosen corruption
                    # This reuses the error handling and severity mapping logic
                    single_corruption_transform = MedMNISTCorruptionTransform(
                        corruption_name=chosen_corruption_name,
                        severity=chosen_severity,
                        corruption_dataset_name=self.collection_source # For context, not strictly used by MedMNISTCorruptionTransform directly
                    )
                    img = single_corruption_transform(img) # This returns PIL image
                    
                    # Ensure consistent size after the corruption
                    # Need to handle cases where single_corruption_transform might return original on error
                    if not isinstance(img, transforms.functional.InterpolationMode): # Check if it's a PIL image
                        resizer = transforms.Compose([
                            transforms.Resize(self.img_size_for_resize),
                            transforms.CenterCrop(self.img_size_for_resize)
                        ])
                        img = resizer(img)
                    return img

            train_transform_list.append(RandomPlainMedMNISTCTransform(collection_source=plain_medmnistc_collection_source, img_size_for_resize=img_size))
        else:
            logging.warning("use_plain_medmnistc is True, but medmnistc library is not available. Plain MedMNISTC will not be applied.")
            train_transform_list.append(transforms.Resize(img_size))
            train_transform_list.append(transforms.CenterCrop(img_size))


    elif use_torchvision_augmix:
        logging.info(f"Applying torchvision.transforms.AugMix with severity={tv_augmix_severity}, mixture_width={tv_augmix_mixture_width}, alpha={tv_augmix_alpha}")
        train_transform_list.append(transforms.AugMix(severity=tv_augmix_severity, mixture_width=tv_augmix_mixture_width, alpha=tv_augmix_alpha))
        train_transform_list.append(transforms.Resize(img_size)) 
        train_transform_list.append(transforms.CenterCrop(img_size))

    elif use_med_augmix:
        if medmnistc_available:
            logging.info(f"Applying MedMNISTCAugMix with dataset: {augmix_corruption_dataset}, severity: {augmix_severity}, width: {augmix_mixture_width}")
            try:
                med_augmix_transform = MedMNISTCAugMix(
                    corruption_dataset_name=augmix_corruption_dataset,
                    severity=augmix_severity,
                    mixture_width=augmix_mixture_width
                )
                train_transform_list.append(med_augmix_transform)
                train_transform_list.append(transforms.Resize(img_size)) 
                train_transform_list.append(transforms.CenterCrop(img_size))
            except Exception as e:
                logging.error(f"Could not initialize MedMNISTCAugMix: {e}. MedAugMix will not be applied.")
                train_transform_list.append(transforms.Resize(img_size))
                train_transform_list.append(transforms.CenterCrop(img_size))
        else:
            logging.warning("use_med_augmix is True, but medmnistc library is not available. MedAugMix will not be applied.")
            train_transform_list.append(transforms.Resize(img_size))
            train_transform_list.append(transforms.CenterCrop(img_size))

    elif augment_train: 
        logging.info("Applying basic augmentations (RandomResizedCrop, RandomHorizontalFlip, ColorJitter).")
        train_transform_list.append(transforms.RandomResizedCrop(img_size))
        train_transform_list.append(transforms.RandomHorizontalFlip())
        train_transform_list.append(transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1))
    else: 
        logging.info("No augmentations applied. Applying Resize and CenterCrop for consistent sizing.")
        train_transform_list.append(transforms.Resize(256)) 
        train_transform_list.append(transforms.CenterCrop(img_size))

    train_transform_list.append(transforms.ToTensor())
    train_transform_list.append(normalize_transform)
    train_transform = transforms.Compose(train_transform_list)

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        normalize_transform,
    ])

    train_dataset = Fitzpatrick17k(
        root=root_dir,
        split='train',
        label_partition=label_partition,
        target_domain=str(target_domain_ood_test), 
        transform=train_transform
    )

    id_val_dataset = Fitzpatrick17k(
        root=root_dir,
        split='val',
        label_partition=label_partition,
        target_domain=str(target_domain_ood_test),
        transform=test_transform
    )

    ood_test_dataset = Fitzpatrick17k(
        root=root_dir,
        split='test',
        label_partition=label_partition,
        target_domain=str(target_domain_ood_test),
        transform=test_transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True 
    )

    id_val_loader = torch.utils.data.DataLoader(
        id_val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    ood_test_loader = torch.utils.data.DataLoader(
        ood_test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, id_val_loader, ood_test_loader
