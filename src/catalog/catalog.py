"""Container class to hold catalog metadata and partition iteration"""

import json
import os

import pandas as pd


class Catalog:
    """Container class for catalog metadata"""

    def __init__(self, catalog_path=None):
        self.catalog_path = catalog_path
        self.metadata_keywords = None
        self.partition_info = None

        self._initialize_metadata()

    def _initialize_metadata(self):
        if not os.path.exists(self.catalog_path):
            raise FileNotFoundError(f"No directory exists at {self.catalog_path}")
        metadata_filename = os.path.join(self.catalog_path, "catalog_info.json")
        if not os.path.exists(metadata_filename):
            raise FileNotFoundError(
                f"No catalog info found where expected: {metadata_filename}"
            )
        partition_info_filename = os.path.join(self.catalog_path, "partition_info.json")
        if not os.path.exists(partition_info_filename):
            raise FileNotFoundError(
                f"No partition info found where expected: {partition_info_filename}"
            )

        with open(metadata_filename, "r", encoding="utf-8") as metadata_info:
            self.metadata_keywords = json.load(metadata_info)
        self.partition_info = pd.read_csv(partition_info_filename)
        ## TODO - pre-fill all the partition file locations?

    def get_pixels(self):
        """Get all healpix pixels that are contained in the catalog

        Returns:
            one-dimensional array of integer 3-tuples
                [0]: order of the destination pixel
                [1]: pixel number *at the above order*
                [2]: the number of rows in the pixel's partition
        """
        ## TODO
        return self.partition_info

    def get_partitions(self):
        """Get file handles for all partition files in the catalog

        Returns:
            one-dimensional array of strings, where each string is a partition file
        """
        ## TODO
        return self.partition_info

    def filter(self):
        """TODO"""
        ## TODO
        return self
