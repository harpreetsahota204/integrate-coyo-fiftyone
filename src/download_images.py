import argparse
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import load_from_disk
import requests
from tqdm.auto import tqdm

def parse_arguments():
    """
    Parses command-line arguments for the image downloading script.

    Returns:
        argparse.Namespace: An object containing the parsed arguments, including the subset path and image directory.
    """
    parser = argparse.ArgumentParser(description='Download images for a specified dataset subset.')
    parser.add_argument('--subset-path', type=Path, required=True,
                        help='Full path to the dataset subset')
    parser.add_argument('--image-directory', type=Path, default=Path('./images'),
                        help='Directory to save downloaded images')
    return parser.parse_args()

def download_image(image_data, image_directory):
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
        response = requests.get(url, stream=True, timeout=10)
        content_type = response.headers.get('Content-Type', '')

        # Check if the response content type is an image
        if 'image' in content_type:
            if response.status_code == 200:
                with open(file_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f"Downloaded {image_id}")
            else:
                print(f"Failed {image_id}: Status code {response.status_code}")
        else:
            print(f"Skipped {image_id}: Content is not an image")
    except Exception as e:
        print(f"Skipped {image_id} due to error: {e}")

def download_images(dataset, image_directory):
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
                result = future.result()
                pbar.update(1)
                print(result)

def main():
    """
    Main function that orchestrates the downloading of images for a dataset subset.
    """
    args = parse_arguments()
    
    # Load the subset from disk
    loaded_subset = load_from_disk(args.subset_path)
    
    # Download images
    download_images(loaded_subset, args.image_directory)

if __name__ == "__main__":
    main()