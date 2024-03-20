# COYO-Tiny FiftyOne Dataset

This repo facilitates the creation of a FiftyOne dataset of the COYO-Tiny project. 

COYO-Tiny is a subset of the [COYO-700M dataset](https://github.com/kakaobrain/coyo-dataset/). 

This subset was created by downloading the full dataset from [Hugging Face](https://huggingface.co/datasets/kakaobrain/coyo-700m). Then a small subset (0.5%) of rows were randomly selected from the full dataset and images were downloaded according to the `download_images.py` script. See that script for more details.

It includes functionality for defining a custom schema, adding metadata and custom fields to samples, and constructing a dataset from a list of image samples.

## Features

- **Custom Schema Definition**: Easily define a custom schema for your FiftyOne dataset to include specific fields relevant to the COYO-Tiny project.

- **Metadata and Custom Fields**: Automatically extract and assign metadata to each sample, along with custom fields such as clip similarity scores, NSFW scores, aesthetic scores, and the number of faces detected in an image.

- **Dataset Construction**: Compile a list of image samples into a FiftyOne dataset, complete with all defined fields and metadata.

## Requirements

Before you can use this tool, ensure you have the following installed:
- Python 3.10
- `fiftyone`
- The `magic` library for MIME type detection
- The `datasets` library for handling dataset loading and manipulation

## Recipe

1. **Get data from Hugging Face (`get_data_from_hf.py`)**: This script will download the entire dataset from Hugging Face. You pass in what percentage you want and it will save a subset. Be sure to delete the full dataset if youd don't want it. The size of the full dataset is 135GB

2. **Download Images (`download_images.py`)**: This script will download images from the provided URLs into your local directory. It does it's best to ensure that only valid images are downloaded.

3. **Filter Downloaded Images (`filter_downloaded_images.py`)**: Once images are downloaded we filter the dataset to include only those images that have *actually* been downloaded. It again creates another `Dataset` whose rows only correspond to tha actual images that were downloaded. You can delete the larger subset after this step.

4. **Create `FiftyOne` dataset** This part happens in the `create_fiftyone_dataset.py` script.  

  - Utilize the `create_coyo_fiftyone_dataset` function to define the schema of your dataset. This includes specifying the fields that will be included in your dataset and their types.

  - **Create FiftyOne Samples**: Use the `create_fo_sample` function to create FiftyOne sample objects for each of your images. This function will automatically assign metadata and any specified custom fields to each sample.

  - **Build Your Dataset**: With your list of FiftyOne samples, call the `add_samples_to_fiftyone_dataset` function to add them to your dataset. This function also supports adding samples dynamically, allowing for flexible dataset construction.

