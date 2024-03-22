import argparse
import os
from pathlib import Path
from datasets import Dataset, load_from_disk
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from typing import Union, List

# Configure logging
def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_arguments():
    """
    Parses command-line arguments for downloading, subsetting, and caching a dataset.

    Returns:
        Namespace: The parsed arguments with dataset name, cache directory, subset percentage, and subset name.
    """
    parser = argparse.ArgumentParser(description='Filters the dataset to include only those images that have been downloaded.')
    parser.add_argument('--subset-path', type=str, required=True, help='Full path to the dataset')
    parser.add_argument('--image-directory', type=str, default=None, help='Directory to downloaded images')
    parser.add_argument('--cache-path', type=str, default=None, required=True, help='Directory to save the cached dataset')
    parser.add_argument('--dataset-name', type=str, default="coyo-tiny", required=True, help='Name of dataset')
    
    args = parser.parse_args()
    
    args.subset_path = Path(args.subset_path)

    # If --image-directory is not specified, assume there is an 'images' directory within --subset-path
    if args.image_directory is None:
        args.image_directory = args.subset_path / 'images'
    else:
        args.image_directory = Path(args.image_directory)
    
    return args

def get_image_paths(images_dir: Path) -> dict:
    """
    Retrieves the paths of all image files within a specified directory.

    This function iterates over all files in the given directory, filtering out non-file entries,
    and constructs a dictionary mapping from image ID (assumed to be the file stem) to the full path of the image.

    Parameters:
        images_dir (Path): The directory containing image files.

    Returns:
        dict: A dictionary where keys are image IDs (int) and values are the full paths (Path objects) to the images.
    """

    image_paths = [f for f in images_dir.iterdir() if f.is_file()]
    return {int(path.stem): path for path in image_paths}

def add_image_path(entry: dict, image_id_to_path: dict) -> bool:
    """
    Adds the path of an image to a dataset entry if it exists in the provided dictionary.

    Args:
        entry (dict): the dataset entry to which the image path will be added.
        image_id_to_path (dict): the dictionary mapping image IDs to their paths.

    Returns:
        bool: a boolean value whether the image path was added to the entry.
    """

    image_id = entry['id']
    if image_id in image_id_to_path:
        entry['image_path'] = str(image_id_to_path[image_id])
        return True
    return False

def parallel_filter_and_add_path(dataset, check_and_add_function, image_id_to_path, max_workers=10):
    """
    Filters a dataset in parallel by applying a check-and-add function to each entry and collects entries for which the function returns True.

    Args:
        dataset (list): A list of dataset entries to be processed.
        check_and_add_function (function): A function that takes two arguments (an entry from the dataset and the image_id_to_path dictionary) and returns True if the entry should be included in the filtered dataset, False otherwise.
        image_id_to_path (dict): The dictionary mapping image IDs to their paths.
        max_workers (int, optional): The maximum number of threads to use for processing.
    """
    
    filtered_dataset_with_paths = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_entry = {executor.submit(check_and_add_function, entry, image_id_to_path): entry for entry in dataset}
        
        for future in as_completed(future_to_entry):
            entry = future_to_entry[future]
            if future.result():
                filtered_dataset_with_paths.append(entry)
                
    return filtered_dataset_with_paths

def cache_dataset(cache_dir: str, dataset: Union[Dataset, List[dict]], dataset_name: str):
    """
    Caches the dataset to disk.

    Args:
        cache_dir (str): The base directory where the dataset will be cached.
        dataset (Union[Dataset, List[dict]]): The dataset to be cached, can be a list of dictionaries or a Dataset object.
        dataset_name (str): The name of the dataset.
    """
    
    try:
        # Define the directory path
        dir_path = Path(cache_dir) / "dataset"
        
        # Create the directory if it doesn't exist
        dir_path.mkdir(parents=True, exist_ok=True)
        
        # Check if dataset needs to be converted to a Dataset object
        if isinstance(dataset, list):
            dataset = Dataset.from_list(dataset)
        
        # Save the dataset to disk
        dataset_path = dir_path / dataset_name
        dataset.save_to_disk(str(dataset_path))
        
        logging.info(f"Dataset '{dataset_name}' cached successfully at {dataset_path}")
    except Exception as e:
        logging.error(f"Failed to cache dataset '{dataset_name}': {e}")

def main():
    setup_logging()
    args = parse_arguments()
    max_workers = os.cpu_count()

    # Load the dataset
    dataset = load_from_disk(args.subset_path)
    
    # Get the paths of all image files
    image_id_to_path = get_image_paths(args.image_directory)
    
    # Filter the dataset and add the image paths
    filtered_dataset_with_paths = parallel_filter_and_add_path(
        dataset, 
        add_image_path, 
        image_id_to_path,  # Pass the dictionary here
        max_workers=max_workers
    )
    # Cache the dataset to disk
    cache_dataset(args.cache_path, filtered_dataset_with_paths, args.dataset_name)

if __name__ == '__main__':
    main()