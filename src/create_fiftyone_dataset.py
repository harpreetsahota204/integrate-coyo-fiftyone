import os
from pathlib import Path
import fiftyone as fo
import fiftyone.core.fields as fof
import magic
from datasets import load_from_disk
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# Constants
DATASET_DIR = "/Users/harpreetsahota/workspace/datasets/coyo_1m_sample"
IMAGES_DIR = Path(DATASET_DIR, "images")
DATASET_NAME = "coyo-tiny"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
    """Adds the path of an image to a dataset entry if it exists in the provided dictionary.

    Args:
        entry (dict): _description_
        image_id_to_path (dict): _description_

    Returns:
        bool: _description_
    """
    image_id = entry['id']
    if image_id in image_id_to_path:
        entry['image_path'] = str(image_id_to_path[image_id])
        return True
    return False


def filter_and_add_paths(dataset: list, image_id_to_path: dict, max_workers: int = os.cpu_count()) -> list:
    """ Filters the dataset and adds the image paths to the entries.    

    Args:
        dataset (list): _description_
        image_id_to_path (dict): _description_
        max_workers (int, optional): _description_. Defaults to os.cpu_count().

    Returns:
        list: _description_
    """
    filtered_dataset_with_paths = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_entry = {executor.submit(add_image_path, entry, image_id_to_path): entry for entry in dataset}
        
        for future in as_completed(future_to_entry):
            entry = future_to_entry[future]
            try:
                if future.result():
                    filtered_dataset_with_paths.append(entry)
            except Exception as e:
                logging.error(f"Error processing entry {entry['id']}: {e}")

    return filtered_dataset_with_paths


def create_fo_sample(image: dict) -> fo.Sample:
    """
    Creates a FiftyOne Sample from a given image entry with metadata and custom fields.

    Args:
        image (dict): A dictionary containing image data including the path and other properties.

    Returns:
        fo.Sample: The FiftyOne Sample object with the image and its metadata.
    """
    sample = fo.Sample(filepath=image['image_path'])
    
    # Attempt to get image size, falling back to 0 if not found.
    try:
        image_size_bytes = os.path.getsize(image['image_path'])
    except FileNotFoundError:
        image_size_bytes = 0
    
    # Attempt to get MIME type, falling back to an empty string if not found.
    try:
        mime_type = magic.from_file(image['image_path'], mime=True)
    except Exception:
        mime_type = ""
    
    # Set the metadata for the sample.
    sample['metadata'] = fo.ImageMetadata(
        height=image.get('height', None),
        width=image.get('width', None),
        size_bytes=image_size_bytes,
        mime_type=mime_type
    )
    
    sample['caption'] = fo.DynamicEmbeddedDocument(
        caption=image['text'],
        text_length = image['text_length'],
        word_count = image['word_count'],
        token_count = image['num_tokens_gpt']
    )
    
    if 'clip_similarity_vitb32' in image:
        sample['clip_similarity_vitb32'] = fo.DynamicEmbeddedDocument(value=image['clip_similarity_vitb32'])
    
    if 'clip_similarity_vitl14' in image:
        sample['clip_similarity_vitl14'] = fo.DynamicEmbeddedDocument(value=image['clip_similarity_vitl14'])
    
    if 'nsfw_score_gantman' in image:
        sample['nsfw_score_gantman'] = fo.DynamicEmbeddedDocument(value=image['nsfw_score_gantman'])
    
    if 'aesthetic_score_laion_v2' in image:
        sample['aesthetic_score_laion_v2'] = fo.DynamicEmbeddedDocument(value=image['aesthetic_score_laion_v2'])
    
    if 'num_faces' in image:
        sample['num_faces'] = fo.DynamicEmbeddedDocument(value=image['num_faces'])
    
    return sample


  def create_fiftyone_dataset(samples: list, dataset_name: str):
      """ Creates a FiftyOne dataset from a list of samples.

      Args:
          samples (list): _description_
          dataset_name (str): _description_
      """
    dataset = fo.Dataset(name=dataset_name, persistent=True)
    dataset.add_samples(samples, dynamic=True)
    dataset.add_dynamic_sample_fields()  

def main():
    """
    Main entry point for the script.
    """
    logging.info("Loading dataset from disk...")
    hf_dataset = load_from_disk(DATASET_DIR)
    
    logging.info("Getting image paths...")
    image_id_to_path = get_image_paths(IMAGES_DIR)
    
    logging.info("Filtering dataset...")
    filtered_dataset_with_paths = filter_and_add_paths(hf_dataset, image_id_to_path)
    
    logging.info("Creating FiftyOne samples...")
    samples = [create_fo_sample(image) for image in filtered_dataset_with_paths]
    
    logging.info("Creating FiftyOne dataset...")
    create_fiftyone_dataset(samples, DATASET_NAME)
    
    logging.info("Dataset creation completed.")

if __name__ == "__main__":
    main()