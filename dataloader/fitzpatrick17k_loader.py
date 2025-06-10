import numpy as np
import torch
import torch.utils
from PIL import Image
import os
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from pathlib import Path


class Fitzpatrick17k(torch.utils.data.Dataset):
    def __init__(self, root, split, label_partition = 3, target_domain = None, transform = None):
        assert label_partition in [3, 9, 114]
        assert split in ['all','train', 'val', 'test'], "Split must be one of ['all','train', 'val', 'test']"

        self.root = root
        self.split = split
        self.label_partition = label_partition 
        self.transform = transform
        self.target_domain = target_domain
        
        self.filepaths = []
        self.labels = []
        self.fiz_scales = []
        
        fpaths, labels, fiz_scales = self._load_images()

        if self.target_domain: # This implies we are doing the specific OOD split
            fpaths, labels, fiz_scales = self._split_data_target(fpaths, labels, fiz_scales)
        elif split in ['train','val','test']: # This is for a general split if target_domain is not specified
            fpaths, labels, fiz_scales = self._split_data(fpaths, labels, fiz_scales)
            
        self.fpaths = fpaths
        self.labels = labels
        self.fiz_scales = fiz_scales


    def __getitem__(self, index):
        img = Image.open(self.fpaths[index]).convert("RGB")
        label = self.labels[index]
        fs = self.fiz_scales[index]

        if self.transform is not None:
            img = self.transform(img)

        return img, label, fs

    def __len__(self):
        return len(self.fpaths)


    def _load_images(self):
        # Corrected CSV filename and image path construction
        metadata_filename = 'fitzpatrick17k.csv' 
        metadata = os.path.join(self.root, metadata_filename)
        image_subfolder = 'patches'

        filepaths, labels_str, fiz_scales = [], [], []

        with open(metadata, 'r') as f:
            # md5hash,fitzpatrick_scale,fitzpatrick_centaur,label,nine_partition_label,three_partition_label,qc,url,url_alphanum
            for line in f.readlines()[1:]:
                fields = line.strip().split(",")

                # Construct path relative to the dataset root
                fpath = os.path.join(self.root, image_subfolder, f'{fields[0]}.jpg')
                if not os.path.isfile(fpath):
                    # print(f"Warning: File not found {fpath}")
                    continue

                if fields[1] in ['1','2']:
                    fiz_scales.append(12) # FSG1
                elif fields[1] in ['3','4']:
                    fiz_scales.append(34) # FSG2
                elif fields[1] in ['5','6']:
                    fiz_scales.append(56) # FSG3
                else: # Skip if FST is unknown or not in 1-6
                    # print(f"Skipping due to FST: {fields[1]} for {fields[0]}")
                    continue 

                filepaths.append(fpath)

                if self.label_partition == 114:
                    labels_str.append(fields[3])
                elif self.label_partition == 3:
                    labels_str.append(fields[5]) # three_partition_label
                elif self.label_partition == 9:
                    labels_str.append(fields[4]) # nine_partition_label

        unique_labels = sorted(list(set(labels_str))) # Ensure unique_labels are actually unique and sorted
        self.map_labels = {lbl:idx for idx,lbl in enumerate(unique_labels)}
        self.num_actual_classes = len(unique_labels) # Store the actual number of classes found
        
        labels = [self.map_labels[lbl] for lbl in labels_str]

        return np.array(filepaths), np.array(labels), np.array(fiz_scales)
    

    def _split_data_target(self, filepaths, labels, domains):
        # Ensure target_domain is treated as an integer if it's passed as a string like '56'
        try:
            target_domain_int = int(self.target_domain)
        except ValueError:
            raise ValueError(f"target_domain '{self.target_domain}' could not be converted to an integer.")

        if self.split == 'test': # This will be our OOD test set (e.g., FSG3)
            mask = domains == target_domain_int
            return filepaths[mask], labels[mask], domains[mask]
        else: # This will be our combined training and validation set from other domains (e.g., FSG1 and FSG2)
            mask = domains != target_domain_int
            trainval_filepaths, trainval_labels, trainval_domains = filepaths[mask], labels[mask], domains[mask]
            
            # Further split the non-target domains into train and validation
            # Stratify by labels within these combined non-target domains
            train_filepaths, val_filepaths, train_labels, val_labels, train_domains, val_domains  = train_test_split(
                trainval_filepaths, trainval_labels, trainval_domains, 
                stratify=trainval_labels, # Stratify by the skin condition labels
                test_size=0.2, # e.g., 20% of (FSG1+FSG2) for validation
                random_state=342234324 # seed independent
            )
            if self.split == 'train':
                return train_filepaths, train_labels, train_domains
            else: # self.split == 'val'
                return val_filepaths, val_labels, val_domains


    def _split_data(self, filepaths, labels, domains):
        # General split: 60% train, 10% val, 30% test from the whole dataset
        # Stratify by primary labels (skin conditions)
        trainval_filepaths, test_filepaths, trainval_labels, test_labels, trainval_domains, test_domains = train_test_split(
            filepaths, labels, domains, test_size=0.3, stratify=labels, random_state=2342342323 # seed indep.
        )
                
        # Split trainval further into train and val
        # test_size for val is 0.1 of original, so 0.1/0.7 of the trainval set
        train_filepaths, val_filepaths, train_labels, val_labels, train_domains, val_domains = train_test_split(
            trainval_filepaths, trainval_labels, trainval_domains, test_size=(0.1 / 0.7), stratify=trainval_labels, random_state=2342342323 # seed indep.
        )
        
        if self.split == 'train':
            return train_filepaths, train_labels, train_domains
        elif self.split == 'val':
            return val_filepaths, val_labels, val_domains
        elif self.split == 'test':
            return test_filepaths, test_labels, test_domains
