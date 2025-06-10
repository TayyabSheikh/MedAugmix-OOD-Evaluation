import torchvision.transforms as transforms
import numpy as np
from PIL import Image, ImageOps, ImageEnhance
import random
import logging

# Attempt to import medmnistc and its corruptions
medmnistc_available = False
ALL_CORRUPTION_CLASSES = {} # Initialize

try:
    # General corruptions from medmnistc.corruptions
    from medmnistc.corruptions import (
        ShotNoise, ImpulseNoise, DefocusBlur, GlassBlur, MotionBlur, ZoomBlur,
        Snow, Frost, Fog, Brightness, Contrast, ElasticTransform, Pixelate, JPEGCompression,
        SpeckleNoise, GaussianBlur, Spatter, Saturate
    )
    general_corruptions_list = [
        ShotNoise, ImpulseNoise, DefocusBlur, GlassBlur, MotionBlur, ZoomBlur,
        Snow, Frost, Fog, Brightness, Contrast, ElasticTransform, Pixelate, JPEGCompression,
        SpeckleNoise, GaussianBlur, Spatter, Saturate
    ]
    for cls in general_corruptions_list:
        ALL_CORRUPTION_CLASSES[cls.__name__] = cls

    # Microscopy-specific corruptions from medmnistc.corruptions.microscopy
    try:
        from medmnistc.corruptions.microscopy import (
            Characters, InkSpill, MarkerPen, Scratches, Stains, WaterDrops, BlackCorner
        )
        microscopy_corruptions_list = [
             Characters, InkSpill, MarkerPen, Scratches, Stains, WaterDrops, BlackCorner
        ]
        for cls in microscopy_corruptions_list:
            ALL_CORRUPTION_CLASSES[cls.__name__] = cls
    except ImportError:
        logging.warning("Some microscopy specific corruptions from medmnistc.corruptions.microscopy not found (e.g., BlackCorner might be newer).")

    medmnistc_available = True
    logging.info(f"Successfully imported medmnistc corruptions. Available classes: {list(ALL_CORRUPTION_CLASSES.keys())}")

except ImportError:
    logging.warning("medmnistc library or its core components not found. MedMNIST-C based augmentations will not be available.")

# Define CORRUPTIONS_DS at the module level
# This list should contain the CLASS NAMES (strings) as keys in ALL_CORRUPTION_CLASSES
# Based on user feedback for the 'dermamnist' set from MedMNIST-C documentation/source.
# Note: 'brightness_up/down' map to 'Brightness', 'contrast_up/down' map to 'Contrast'.
# The medmnistc library handles directionality within the .apply() method based on severity.
DERMAMNIST_CORRUPTION_CLASS_NAMES = [
    "Pixelate",
    "JPEGCompression", # Corresponds to 'jpeg_compression'
    "GaussianNoise",   # Corresponds to 'gaussian_noise'
    "SpeckleNoise",    # Corresponds to 'speckle_noise'
    "ImpulseNoise",    # Corresponds to 'impulse_noise'
    "ShotNoise",       # Corresponds to 'shot_noise'
    "DefocusBlur",     # Corresponds to 'defocus_blur'
    "MotionBlur",      # Corresponds to 'motion_blur'
    "ZoomBlur",        # Corresponds to 'zoom_blur'
    "Brightness",      # Covers 'brightness_up' and 'brightness_down'
    "Contrast",        # Covers 'contrast_up' and 'contrast_down'
    "Characters",      # Corresponds to 'characters'
    "BlackCorner"      # Corresponds to 'black_corner' 
                        # (Ensure BlackCorner class was successfully imported and is in ALL_CORRUPTION_CLASSES)
]

CORRUPTIONS_DS = {
    'dermamnist': DERMAMNIST_CORRUPTION_CLASS_NAMES,
    'bloodmnist': [ # Example, kept from previous
        "GaussianNoise", "ShotNoise", "ImpulseNoise", "Spatter", "GaussianBlur", 
        "SpeckleNoise", "Contrast", "Brightness", "Saturate"
    ] 
    # Add other collections like 'pathmnist' if needed
}

# Filter CORRUPTIONS_DS to only include available corruption classes that were successfully imported
for ds_name, corr_name_list in CORRUPTIONS_DS.items():
    available_names_for_ds = []
    for name in corr_name_list:
        if name in ALL_CORRUPTION_CLASSES:
            available_names_for_ds.append(name)
        else:
            logging.warning(f"Corruption class name '{name}' for dataset '{ds_name}' not found in ALL_CORRUPTION_CLASSES. It will be excluded.")
    CORRUPTIONS_DS[ds_name] = available_names_for_ds
    logging.info(f"Final available corruption class names for {ds_name}: {CORRUPTIONS_DS[ds_name]}")


def get_medmnistc_corruption_names_by_collection(collection_name="dermamnist"):
    """
    Returns a list of available corruption class names (strings) for a given MedMNIST-C collection.
    """
    return CORRUPTIONS_DS.get(collection_name, [])


class MedMNISTCorruptionTransform:
    def __init__(self, corruption_name, severity, corruption_dataset_name="dermamnist"):
        self.corruption_name = corruption_name
        self.severity = severity 
        self.corruption_dataset_name = corruption_dataset_name 

        if not medmnistc_available:
            raise ImportError("medmnistc library is required for MedMNISTCorruptionTransform.")

        CorruptionClass = ALL_CORRUPTION_CLASSES.get(self.corruption_name)
        if CorruptionClass is None:
            raise ValueError(f"Unknown or unavailable MedMNIST-C corruption class: {self.corruption_name}")
        
        self.corruption_instance = CorruptionClass()

    def __call__(self, img): 
        original_input_pil_image = img.copy() 

        try:
            if isinstance(self.corruption_instance, ImpulseNoise) and not hasattr(self.corruption_instance, 'rng'):
                logging.debug(f"Patching 'rng' attribute for ImpulseNoise instance ({self.corruption_instance}) before calling apply.") # Changed to debug
                self.corruption_instance.rng = np.random.default_rng()
            
            current_severity = self.severity
            # MedMNIST-C corruptions generally expect severity 1-5.
            # ZoomBlur is an exception, expecting 0-4 for its internal severity_params indexing.
            if isinstance(self.corruption_instance, ZoomBlur): 
                current_severity = max(0, self.severity - 1) 

            corruption_output = self.corruption_instance.apply(img, severity=current_severity)
            
            if corruption_output is None:
                logging.warning(f"{self.corruption_name}.apply returned None, returning original image.")
                return original_input_pil_image

            if not isinstance(corruption_output, Image.Image):
                if isinstance(corruption_output, np.ndarray):
                    if corruption_output.dtype != np.uint8: 
                        corruption_output = np.clip(corruption_output, 0, 255).astype(np.uint8)
                    corrupted_pil_img = Image.fromarray(corruption_output)
                else: 
                    logging.warning(f"{self.corruption_name}.apply did not return PIL Image or ndarray, returning original. Type: {type(corruption_output)}")
                    return original_input_pil_image
            else:
                corrupted_pil_img = corruption_output
            
            return corrupted_pil_img

        except ValueError as ve:
            if isinstance(self.corruption_instance, ZoomBlur) and "MAX_ZOOM" in str(ve):
                logging.warning(f"Caught ValueError in ZoomBlur (MAX_ZOOM issue): {ve}. Returning original image.")
            elif "Characters" == self.corruption_name and "empty range in randrange" in str(ve):
                logging.warning(f"Caught ValueError in Characters (image too small): {ve}. Returning original image.")
            else: 
                logging.error(f"ValueError during {self.corruption_name}.apply: {ve}", exc_info=False) 
            return original_input_pil_image
        except Exception as e: 
            logging.error(f"Error during {self.corruption_name}.apply or processing: {e}", exc_info=True)
            return original_input_pil_image


class MedMNISTCAugMix(object):
    def __init__(self, corruption_dataset_name="dermamnist", severity=3, mixture_width=3, alpha=1.0, img_size=224):
        self.corruption_dataset_name = corruption_dataset_name
        self.severity = severity
        self.mixture_width = mixture_width
        self.alpha = alpha
        self.img_size = img_size 

        if not medmnistc_available:
            raise ImportError("medmnistc library is required for MedMNISTCAugMix.")

        self.corruption_names_for_augmix = get_medmnistc_corruption_names_by_collection(self.corruption_dataset_name)
        
        if not self.corruption_names_for_augmix:
            raise ValueError(f"No MedMNIST-C operations available for {corruption_dataset_name} for AugMix (list is empty after filtering).")

        corruptions_to_skip_in_augmix = ["MotionBlur", "ZoomBlur", "Characters"] 
        
        self.corruption_ops = []
        skipped_count = 0
        for name in self.corruption_names_for_augmix: 
            if name in corruptions_to_skip_in_augmix:
                logging.debug(f"Skipping {name} for MedMNISTCAugMix as it's in the explicit AugMix skip list.") # Changed to debug
                skipped_count +=1
                continue
            try:
                # Create a transform instance for each op to be used in AugMix
                self.corruption_ops.append(MedMNISTCorruptionTransform(name, self.severity, self.corruption_dataset_name))
            except ValueError as e: 
                 logging.warning(f"Could not create MedMNISTCorruptionTransform for '{name}': {e}. Skipping for AugMix.")
                 skipped_count +=1

        if not self.corruption_ops:
            logging.warning(f"No corruption operations available for MedMNISTCAugMix with dataset {self.corruption_dataset_name} after filtering. AugMix may not apply any corruptions.")

        logging.info(f"MedMNISTCAugMix initialized with {len(self.corruption_ops)} operations for {self.corruption_dataset_name} (skipped {skipped_count}). Severity: {self.severity}, Width: {self.mixture_width}, Alpha: {self.alpha}")


    def __call__(self, img): 
        if not self.corruption_ops: 
            return img

        mixture_weights = np.random.dirichlet([self.alpha] * self.mixture_width)
        mixed_img_accumulator = np.zeros_like(np.array(img, dtype=np.float32))

        for i in range(self.mixture_width):
            op = random.choice(self.corruption_ops)
            corrupted_img_pil = op(img.copy()) 
            
            corrupted_img_np = np.array(corrupted_img_pil, dtype=np.float32)
            mixed_img_accumulator += mixture_weights[i] * corrupted_img_np
        
        original_img_np = np.array(img, dtype=np.float32)
        final_mixed_img_np = (1 - np.sum(mixture_weights)) * original_img_np + mixed_img_accumulator
        
        mixed_img_clipped = np.clip(final_mixed_img_np, 0, 255).astype(np.uint8)
        return Image.fromarray(mixed_img_clipped)
