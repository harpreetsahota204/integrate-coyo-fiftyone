import logging
import os
import shutil

import eta.core.serial as etas
import eta.core.utils as etau
import eta.core.web as etaw

import fiftyone.types as fot
import fiftyone.utils.data as foud

from fiftyone.zoo import ZooDataset, ZooDatasetInfo

logger = logging.getLogger(__name__)

class CoyoTinyDataset(ZooDataset):
    @property
    def name(self):
        return "COYO-Tiny"
        
    @property
    def tags(self):
        """A tuple of tags for the dataset."""
        return ("image", "image-text", "captions")

    @property
    def parameters(self) -> None:
        """An optional dict of parameters describing the configuration of the
        zoo dataset when it was downloaded.
        """
        return None

    @property
    def supported_splits(self) -> None:
        """A tuple of supported splits for the dataset, or None if the dataset
        does not have splits.
        """
        return None

    @property
    def importer_kwargs(self):
        """A dict of default kwargs to pass to this dataset's
        :class:`fiftyone.utils.data.importers.DatasetImporter`.
        """
        return {}

    def _download_and_prepare(self, dataset_dir, scratch_dir, split):
        """Internal implementation of downloading the dataset and preparing it
        for use in the given directory.

        Args:
            dataset_dir: the directory in which to construct the dataset
            scratch_dir: a scratch directory to use to download and prepare
                any required intermediate files
            split: the split to download, or None if the dataset does not have
                splits

        Returns:
            tuple of

            -   dataset_type: the :class:`fiftyone.types.Dataset` type of the
                dataset
            -   num_samples: the number of samples in the split. For datasets
                that support partial downloads, this can be ``None``, which
                indicates that all content was already downloaded
            -   classes: an optional list of class label strings
        """
        raise NotImplementedError(
            "subclasses must implement _download_and_prepare()"
        )

    def _patch_if_necessary(self, dataset_dir, split):
        """Internal method called when an already downloaded dataset may need
        to be patched.

        Args:
            dataset_dir: the directory containing the dataset
            split: the split to patch, or None if the dataset does not have
                splits
        """
        raise NotImplementedError(
            "subclasses must implement _patch_if_necessary()"
        )


