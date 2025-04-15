import wilds
import argparse
import os

# Ensure the target directory exists
parser = argparse.ArgumentParser(description='Download Camelyon17 dataset using WILDS.')
parser.add_argument('--root_dir', default='./data', type=str, help='Root directory for WILDS datasets.')
args = parser.parse_args()

print(f"Attempting to download Camelyon17 to: {os.path.abspath(args.root_dir)}")

try:
    # Get the dataset object. This will trigger download if data is not found.
    dataset = wilds.get_dataset(dataset='camelyon17', root_dir=args.root_dir, download=True)
    print("Camelyon17 dataset download check completed.")
    print(f"Dataset location: {dataset.root}")
    # You can optionally print dataset info
    # print(dataset)
except Exception as e:
    print(f"An error occurred during dataset download/check: {e}")
