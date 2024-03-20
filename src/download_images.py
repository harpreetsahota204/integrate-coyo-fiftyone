import argparse
import os
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import Dataset, load_from_disk
import requests
from tqdm.auto import tqdm
import shutil

def parse_arguments():
    """
    Parses command-line arguments for the image downloading script.

    Returns:
        argparse.Namespace: An object containing the parsed arguments, including the subset path and image directory.
    """
    parser = argparse.ArgumentParser(description='Download images for a specified dataset subset.')
    parser.add_argument('--subset-path', type=str, required=True, help='Full path to the dataset subset')
    parser.add_argument('--image-directory', type=str, default=None, help='Directory to save downloaded images')
    args = parser.parse_args()
    args.subset_path = Path(args.subset_path)

    # If --image-directory is not specified, create an 'images' directory within --subset-path
    if args.image_directory is None:
        args.image_directory = args.subset_path / 'images'
    else:
        args.image_directory = Path(args.image_directory)

    # Create the directory if it doesn't exist
    args.image_directory.mkdir(parents=True, exist_ok=True)
    
    return args

def setup_logging():
    logging.basicConfig(filename='download_images.log', level=logging.INFO,
                        format='%(asctime)s %(levelname)s:%(message)s')


def download_image(image_data:dict, image_directory:Path):
    """
    Attempts to download an image from a URL and saves it to the specified directory.
    Skips the download on any error or if the content is not an image.

    Parameters:
        image_data (dict): A dictionary containing the 'url' and 'id' of the image.
        image_directory (Path): The directory where the image will be saved.
    """
    url = image_data['url']
    image_id = image_data['id']
    file_path = image_directory / f"{image_id}.jpg"

    try:
        response = requests.get(url, stream=True, timeout=5)
        content_type = response.headers.get('Content-Type', '').lower()

        if 'image' in content_type:
            if response.status_code == 200:
                with open(file_path, 'wb') as f:
                    shutil.copyfileobj(response.raw, f)
                return f"Downloaded {image_id}"
            else:
                return f"Failed {image_id}: Status code {response.status_code}"
        else:
            return f"Skipped {image_id}: Content is not an image"
    except Exception as e:
        return f"Skipped {image_id} due to error: {e}"

def download_images(dataset:Dataset, image_directory:Path):
    """
    Downloads all images in the dataset using multiple threads.

    Parameters:
        dataset (Dataset): The dataset from which to download images.
        image_directory (Path): The directory where images will be saved.
    """
    num_workers = os.cpu_count()
    image_directory.mkdir(parents=True, exist_ok=True)
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(download_image, image_data, image_directory) for image_data in dataset]
        
        with tqdm(total=len(futures), desc="Downloading images") as pbar:
            for future in as_completed(futures):
                message = future.result()
                pbar.update(1)
                logging.info(message)



def main():
    """
    Main function that orchestrates the downloading of images for a dataset subset.
    """
    setup_logging()
    args = parse_arguments()
    
    loaded_subset = load_from_disk(args.subset_path)
    dataset_size = len(loaded_subset)
    
    user_confirmation = input(f"This will download {dataset_size:,} images. Are you sure you want to do this? (yes/no): ").strip().lower()
    if user_confirmation == 'no':
        choice = input("Enter '%' to download a percentage of the images, or 'n' to specify a number of images: ").strip().lower()
        if choice == '%':
            percentage = float(input("Enter the percentage of images to download (0-100): "))
            subset_size = int((percentage / 100) * dataset_size)
        elif choice == 'n':
            subset_size = int(input("Enter the number of images to download: "))
        else:
            print("Invalid choice. Exiting.")
            
        # Shuffle the dataset and select a subset
        loaded_subset = loaded_subset.shuffle(seed=42).select(range(subset_size))
        print(f"Proceeding to download {subset_size:,} images.")
    elif user_confirmation == 'yes':
        subset_size = dataset_size
        print(f"Proceeding to download {subset_size:,} images.")
    else:
        print("Invalid input. Exiting.")
        
    
    download_images(loaded_subset, args.image_directory)

if __name__ == "__main__":
    main()