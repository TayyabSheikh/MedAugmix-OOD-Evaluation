# Evaluating OOD Generalization with MedAugmix

This repository contains the official source code for the Master's Thesis: "Evaluating Hyperspherical Embeddings and Targeted Augmentations for Generalizable Medical Image Analysis" by Muhammad Tayyab Sheikh.

## About The Project

This project systematically benchmarks Out-of-Distribution (OOD) generalization strategies across two distinct medical domains: histopathology (WILDS Camelyon17) and dermatology (Fitzpatrick17k). It compares the specialized representation learning algorithm HYPO against a standard ERM baseline.

The key contribution is the proposal and evaluation of **MedAugmix**, a novel data augmentation strategy that adapts the AugMix framework to use targeted, clinically-relevant corruptions from MedMNIST-C. Our findings show that a simple ERM model, when enhanced with MedAugmix, consistently outperforms the specialized HYPO algorithm.

### Built With
* [PyTorch](https://pytorch.org/)
* [WILDS Benchmark](https://wilds.stanford.edu/)
* [MedMNIST-C](https://github.com/francescodisalvo05/medmnistc-api)

## Getting Started

To get a local copy up and running, follow these simple steps.

### Prerequisites

You will need Python 3.9+ and pip installed. This project uses CUDA for GPU acceleration.

### Installation

1.  Clone the repo:
    ```sh
    git clone [https://github.com/your_username/your_repository_name.git](https://github.com/your_username/your_repository_name.git)
    cd your_repository_name
    ```
2.  Install the required Python packages:
    ```sh
    pip install -r requirements.txt
    ```
3.  Download the datasets (e.g., WILDS Camelyon17) into a `/data` directory (which is ignored by git).

## Usage

To replicate the key experiments, you can use the provided training scripts. For example, to run ERM with MedAugmix on Camelyon17:

```sh
python train_erm_medmnistc.py --model densenet121 --use_med_augmix --augmix_severity 5 --augmix_mixture_width 1 --batch_size 384 --wilds_root_dir ./data
