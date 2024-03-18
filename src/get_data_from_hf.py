import argparse
from datasets import load_dataset
from pathlib import Path

def parse_arguments():
    """
    Parses command-line arguments for downloading, subsetting, and caching a dataset.

    Returns:
        Namespace: The parsed arguments with dataset name, cache directory, subset percentage, and subset name.
    """
    parser = argparse.ArgumentParser(description='Download, subset, and cache a dataset from Hugging Face.')
    parser.add_argument('--dataset-name', type=str, default='facebook/winoground',
                        help='Name of the dataset to download')
    parser.add_argument('--cache-dir', type=Path, default=Path('./datasets_cache'),
                        help='Directory to cache the downloaded datasets')
    parser.add_argument('--subset-percentage', type=float, default=10.0,
                        help='Percentage of the dataset to keep after shuffling')
    parser.add_argument('--subset-name', type=str, default='subset',
                        help='Name for the saved subset')
    return parser.parse_args()

def main():
    """
    Main function to download a dataset from Hugging Face, shuffle it, select a subset, and save the subset to disk.
    """
    args = parse_arguments()
    
    # Ensure cache directory exists
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Load the dataset
    dataset = load_dataset(
        args.dataset_name, 
        cache_dir=args.cache_dir,
        split="train")
    
    
    # Shuffle and select a subset
    subset_size = int(len(dataset) * (args.subset_percentage / 100))
    subset = dataset.shuffle(seed=42).select(range(subset_size))
    
    # Save the subset to disk
    subset_path = args.cache_dir / args.subset_name
    subset.save_to_disk(subset_path)

if __name__ == "__main__":
    main()