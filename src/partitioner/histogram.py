"""Utilities for generating and manipulating object count histograms"""

import healpy as hp
import numpy as np

from partitioner.io_utils import read_dataframe


def empty_histogram(highest_order):
    """Use numpy to create an histogram array with the right shape, filled with zeros.
    Args:
        highest_order (int): the highest healpix order (e.g. 0-10)
    Returns:
        one-dimensional numpy array of long integers, where the length is equal to
        the number of pixels in a healpix map of target order, and all values are set to 0.
    """
    return np.zeros(hp.order2npix(highest_order), dtype=np.ulonglong)


def generate_partial_histogram(
    file_path,
    file_format,
    highest_order,
    ra_column,
    dec_column,
):
    """Generate a histogram of counts for objects found in the indicated file_path

    Args:
        file_path (str): full path to the input file
        file_format (str): expected format for the input file. See io_utils.read_dataframe
            for accepted formats.
        highest_order (int):  the highest healpix order (e.g. 0-10)
        ra_column (str): where in the input to find the celestial coordinate, right ascension
        dec_column (str): where in the input to find the celestial coordinate, declination
    Returns:
        one-dimensional numpy array of long integers where the value at each index corresponds
        to the number of objects found at the healpix pixel.
    Raises:
        ValueError: if the `ra_column` or `dec_column` cannot be found in the input file.
        FileNotFoundError: See io_utils.read_dataframe for other error conditions.
    """
    histo = empty_histogram(highest_order)
    data = read_dataframe(file_path, file_format)

    required_columns = [ra_column, dec_column]

    # Verify that the file has columns with desired names.
    if not all(x in data.columns for x in required_columns):
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
