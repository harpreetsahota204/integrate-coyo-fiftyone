import argparse
import os
from datasets import load_dataset
from pathlib import Path

def parse_arguments():
    """
    Parses command-line arguments for downloading, subsetting, and caching a dataset.

    Returns:
        Namespace: The parsed arguments with dataset name, cache directory, subset percentage, and subset name.
    """
    parser = argparse.ArgumentParser(description='Download, subset, and cache a dataset from Hugging Face.')
    parser.add_argument('--dataset-name', type=str, default='kakaobrain/coyo-700m',
                        help='Name of the dataset to download')
    parser.add_argument('--cache-dir', type=Path, default=Path('./datasets_cache'),
                        help='Directory to cache the downloaded datasets')
    parser.add_argument('--subset-percentage', type=float, default=1.0,
                        help='Percentage of the dataset to keep after shuffling')
    parser.add_argument('--subset-name', type=str, default='coyo-tiny',
                        help='Name for the saved subset')
    return parser.parse_args()

def main():
    """
    Main function to download a dataset from Hugging Face, shuffle it, select a subset, and save the subset to disk.
    """
    args = parse_arguments()
    
    # Ensure cache directory exists
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading and caching {args.dataset_name} to {args.cache_dir}")
    # Load the dataset
    dataset = load_dataset(
        args.dataset_name, 
        cache_dir=args.cache_dir,
        split="train")
    print("Dataset downloaded and cached")
    
    # Shuffle and select a subset
    print(" Shuffling and selecting a subset...")
    subset_size = int(len(dataset) * (args.subset_percentage / 100))
    subset = dataset.shuffle(seed=42).select(range(subset_size))
    
    # Save the subset to disk
    print(f"Saving subset to {args.cache_dir / args.subset_name}")
    subset_path = args.cache_dir / args.subset_name
    subset.save_to_disk(
        dataset_path = subset_path,
        num_proc=os.cpu_count()
        )
if __name__ == "__main__":
    main()