"""Utilities for generating and manipulating object count histograms"""

import healpy as hp
import numpy as np

from partitioner.io_utils import read_dataframe


def empty_histogram(highest_order):
    """Use numpy to create an histogram array with the right shape, filled with zeros"""
    return np.zeros(hp.order2npix(highest_order), dtype=np.ulonglong)


def generate_partial_histogram(
    file_path,
    highest_order,
    file_format,
    ra_column,
    dec_column,
):
    """Generate a histogram of counts for objects found in the indicated file_path"""
    histo = empty_histogram(highest_order)
    data = read_dataframe(file_path, file_format)

    # Verify that the file has columns with desired names.
    if not all([x in data.columns for x in [ra_column, dec_column]]):
        raise ValueError(
            f"Invalid column names in input file: {ra_column}, {dec_column} not in {file_path}"
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
    return histo
