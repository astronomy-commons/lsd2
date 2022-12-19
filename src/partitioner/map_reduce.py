"""Utilities for generating and manipulating object count histograms"""

import logging
import os

import healpy as hp
import numpy as np

from partitioner.histogram import empty_histogram
from partitioner.io_utils import concatenate_parquet_files, read_dataframe

LOGGER = logging.getLogger(__name__)


def index_is_range_like(data_index):
    """Test if a data frame's index is 'range-like'

    This means:
      - integers
      - monotonically increasing
      - no duplicates
      - span from [0, len(data))
    """
    if not data_index.is_integer():
        LOGGER.warning("Data index is non-integer. Re-indexing.")
        return False
    if data_index.has_duplicates:
        LOGGER.warning("Data index contains duplicates. Re-indexing.")
        return False
    if data_index.hasnans:
        LOGGER.warning("Data index contains nans. Re-indexing.")
        return False
    if data_index.min() != 0:
        LOGGER.warning("Data index doesn't start at 0. Re-indexing.")
        return False
    if data_index.max() != len(data_index) - 1:
        LOGGER.warning("Data index larger than length of data. Re-indexing.")
        return False

    return True


def map_to_pixels(
    input_file,
    file_format,
    highest_order,
    ra_column,
    dec_column,
    shard_index,
    cache_path,
):
    """Map a file if input objects to their healpix pixels.

    Copy objects into pixel-specific parquet files and return histogram
    of counts for objects found in the indicated input_file

    Args:
        input_file (str): fully-specified path to an input file
        file_format (str): expected format for the input file. See io_utils.read_dataframe
            for accepted formats.
        highest_order (int):  the highest healpix order (e.g. 0-10)
        ra_column (str): where in the input to find the celestial coordinate, right ascension
        dec_column (str): where in the input to find the celestial coordinate, declination
        shard_index (int): unique number for this shard of mapped data.
            if mapping from multiple input files, this can be the index
            in the list of input files.
        cache_path (str): directory where temporary pixel parquet files
            will be written

    Returns:
        one-dimensional numpy array of long integers where the value at each index corresponds
        to the number of objects found at the healpix pixel.
    Raises:
        ValueError: if the `ra_column` or `dec_column` cannot be found in the input file.
        FileNotFoundError: See io_utils.read_dataframe for other error conditions.
    """
    histo = empty_histogram(highest_order)
    data = read_dataframe(input_file, file_format)

    if not index_is_range_like(data.index):
        LOGGER.warning("Reindexing data can be expensive.")
        data.reset_index(inplace=True)
        LOGGER.info("Reindexing complete")

    # Verify that the file has columns with desired names.
    if not all([x in data.columns for x in [ra_column, dec_column]]):
        raise ValueError(
            f"Invalid column names in input file: {ra_column}, {dec_column} not in {input_file}"
        )
    mapped_pixels = hp.ang2pix(
        2**highest_order,
        data[ra_column].values,
        data[dec_column].values,
        lonlat=True,
        nest=True,
    )
    mapped_pixel, count_at_pixel = np.unique(mapped_pixels, return_counts=True)
    histo[mapped_pixel] += count_at_pixel.astype(np.ulonglong)

    for pixel in mapped_pixel:
        data_indexes = np.where(mapped_pixels == pixel)
        filtered_data = data.filter(items=data_indexes[0].tolist(), axis=0)

        pixel_dir = os.path.join(cache_path, f"pixel_{pixel}")
        os.makedirs(pixel_dir, exist_ok=True)
        output_file = os.path.join(pixel_dir, f"shard_{shard_index}.parquet")
        filtered_data.to_parquet(output_file)
    return histo


def reduce_shards(
    cache_path,
    origin_pixel_numbers,
    destination_pixel_order,
    destination_pixel_number,
    destination_pixel_size,
    output_path,
    id_column,
):
    """something"""
    destination_dir = os.path.join(
        output_path,
        f"Norder{int(destination_pixel_order)}/Npix{int(destination_pixel_number)}",
    )
    os.makedirs(destination_dir, exist_ok=True)

    destination_file = os.path.join(destination_dir, "catalog.parquet")

    input_directories = []

    for pixel in origin_pixel_numbers:
        pixel_dir = os.path.join(cache_path, f"pixel_{pixel}")
        input_directories.append(pixel_dir)

    rows_written = concatenate_parquet_files(
        input_directories=input_directories,
        output_file_name=destination_file,
        sorting=id_column,
    )

    if rows_written != destination_pixel_size:
        raise ValueError(
            "Unexpected number of objects at pixel "
            f"({destination_pixel_order}, {destination_pixel_number})."
            f" Expected {destination_pixel_size}, wrote {rows_written}"
        )
