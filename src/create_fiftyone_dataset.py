"""
Creates a local fiftyone dataset from a COYO-Tiny dataset.
"""
import argparse
import os
import magic
import logging
import fiftyone as fo
import fiftyone.core.fields as fof

from pathlib import Path
from datasets import Dataset, load_from_disk

from concurrent.futures import ThreadPoolExecutor, as_completed

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
    parser.add_argument('--dataset-name', type=str, default="coyo-tiny", required=True, help='Name of dataset')
    
    args = parser.parse_args()
    
    args.subset_path = Path(args.subset_path)
	
    args.dataset_path = args.subset_path / args.dataset_name

    # If --image-directory is not specified, assume there is an 'images' directory within --subset-path
    if args.image_directory is None:
        args.image_directory = args.subset_path / 'images'
    else:
        args.image_directory = Path(args.image_directory)
    
    return args


def create_coyo_fiftyone_dataset(name) -> fo.Dataset:
	"""
	Creates schema for a COYO-Tiny FiftyOne dataset.
	"""
	dataset = fo.Dataset(name=name, persistent=True, overwrite=True)

	dataset.add_sample_field('image_path', fof.StringField)

	dataset.add_sample_field(
		'clip_similarity_vitb32', 
		fof.FloatField, 
		description='The cosine similarity between text and image(ViT-B/32) embeddings by OpenAI CLIP'
	)
	
	dataset.add_sample_field(
		'clip_similarity_vitl14', 
		fof.FloatField, 
		description='The cosine similarity between text and image(ViT-L/14) embeddings by OpenAI CLIP'
	)

	dataset.add_sample_field(
		'nsfw_score_gantman', 
		fof.FloatField, 
		description='The NSFW score of the image by GantMan/NSFW'
	)

	dataset.add_sample_field(
		'nsfw_score_opennsfw2', 
		fof.FloatField, 
		description='The NSFW score of the image by OpenNSFW2'
	)
		
	dataset.add_sample_field(
		'aesthetic_score_laion_v2', 
		fof.FloatField, 
		description='The aesthetic score of the image by Laion V2'
	)

	dataset.add_sample_field(
		'num_faces', 
		fof.IntField, 
		description='The number of faces in the image detected by SCRFD'
	)
	
	return dataset

def create_fo_sample(image: dict) -> fo.Sample:
	"""
	Creates a FiftyOne Sample from a given image entry with metadata and custom fields.

	Args:
		image (dict): A dictionary containing image data including the path and other properties.

	Returns:
		fo.Sample: The FiftyOne Sample object with the image and its metadata.
	"""
	filepath = image['image_path']
	
	# Attempt to get image size, falling back to 0 if not found.
	try:
		image_size_bytes = os.path.getsize(filepath)
	except FileNotFoundError:
		image_size_bytes = 0
	
	# Attempt to get MIME type, falling back to an empty string if not found.
	try:
		mime_type = magic.from_file(filepath, mime=True)
	except Exception:
		mime_type = ''
	
	# Set the metadata for the sample.
	metadata = fo.ImageMetadata(
		height=image.get('height', None),
		width=image.get('width', None),
		size_bytes=image_size_bytes,
		mime_type=mime_type
	)
	
	caption = fo.DynamicEmbeddedDocument(
		caption=image['text'],
		text_length = image['text_length'],
		word_count = image['word_count'],
		token_count_gpt = image['num_tokens_gpt'],
		token_count_bert = image['num_tokens_bert']
	)
	
	if 'clip_similarity_vitb32' in image:
		clip_similarity_vitb32 = image['clip_similarity_vitb32']
	else:
		clip_similarity_vitb32 = None
	
	if 'clip_similarity_vitl14' in image:
		clip_similarity_vitl14 = image['clip_similarity_vitl14']
	else:
		clip_similarity_vitl14 = None
		
	if 'nsfw_score_gantman' in image:
		nsfw_score_gantman = image['nsfw_score_gantman']
	else:
		nsfw_score_gantman = None

	if 'nsfw_score_opennsfw2' in image:
		nsfw_score_opennsfw2 = image['nsfw_score_gantman']
	else:
		nsfw_score_opennsfw2 = None

	if 'aesthetic_score_laion_v2' in image:
		aesthetic_score_laion_v2 = image['aesthetic_score_laion_v2']
	else:
		aesthetic_score_laion_v2 = None
	
	if 'num_faces' in image:
		num_faces = image['num_faces']
	else:
		num_faces = None

	sample = fo.Sample(
		filepath=filepath,
		metadata=metadata,
		caption=caption,
		clip_similarity_vitb32=clip_similarity_vitb32,
		clip_similarity_vitl14=clip_similarity_vitl14,
		nsfw_score_gantman=nsfw_score_gantman,
		nsfw_score_opennsfw2=nsfw_score_opennsfw2,
		aesthetic_score_laion_v2=aesthetic_score_laion_v2,
		num_faces=num_faces	
	)

	return sample

def add_samples_to_fiftyone_dataset(
	dataset: fo.Dataset,
	samples: list
	):
	"""
	Creates a FiftyOne dataset from a list of samples.

	Args:
		samples (list): _description_
		dataset_name (str): _description_
	"""
	dataset.add_samples(samples, dynamic=True)
	dataset.add_dynamic_sample_fields()
  
def main():
	"""
	Main entry point for the script.
	"""
	setup_logging()
	args = parse_arguments()
	
	logging.info('Creating FiftyOne dataset...')
	dataset = create_coyo_fiftyone_dataset(args.dataset_name)

	logging.info('Loading dataset from disk...')
	hf_dataset = load_from_disk(args.dataset_path)

	logging.info('Creating FiftyOne samples...')
	samples = [create_fo_sample(image) for image in hf_dataset]

	logging.info('Adding samples to FiftyOne ...')
	add_samples_to_fiftyone_dataset(dataset, samples)

	logging.info('Dataset creation completed.')


if __name__ == '__main__':
	main()