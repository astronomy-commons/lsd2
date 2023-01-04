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


def generate_alignment(histogram, highest_order=10, threshold=1_000_000):
    """Generate alignment from high order pixels to those of equal or lower order

    Note:
        We may initially find healpix pixels at order 10, but after aggregating up to the pixel
        threshold, some final pixels are order 4 or 7. This method provides a map from pixels
        at order 10 to their destination pixel. This may be used as an input into later partitioning
        map reduce steps.
    Args:
        histogram (:obj:`np.array`): one-dimensional numpy array of long integers where the
            value at each index corresponds to the number of objects found at the healpix pixel.
        highest_order (int):  the highest healpix order (e.g. 0-10)
        threshold (int): the maximum number of objects allowed in a single pixel
    Returns:
        one-dimensional numpy array of integer 3-tuples, where the value at each index corresponds
        to the destination pixel at order less than or equal to the `highest_order`.
        The tuple contains three integers:
            [0]: order of the destination pixel
            [1]: pixel number *at the above order*
            [2]: the number of objects in the pixel
    """

    if len(histogram) != hp.order2npix(highest_order):
        raise ValueError("histogram is not the right size")

    nested_sums = []
    for i in range(0, highest_order):
        nested_sums.append(empty_histogram(i))
    nested_sums.append(histogram)

    # work backward - from highest order, fill in the sums of lower order pixels
    for read_order in range(highest_order, 0, -1):
        parent_order = read_order - 1
        for index in range(0, len(nested_sums[read_order])):
            parent_pixel = index >> 2
            nested_sums[parent_order][parent_pixel] += nested_sums[read_order][index]

    nested_alignment = []
    for i in range(0, highest_order + 1):
        nested_alignment.append(np.full(hp.order2npix(i), None))

    # work forward - determine if we should map to a lower order pixel, this pixel, or keep looking.
    for read_order in range(0, highest_order + 1):
        parent_order = read_order - 1
        for index in range(0, len(nested_sums[read_order])):
            parent_alignment = None
            if parent_order >=0:
                parent_pixel = index >> 2
                parent_alignment = nested_alignment[parent_order][parent_pixel]

            if parent_alignment:
                nested_alignment[read_order][index] = parent_alignment
            elif nested_sums[read_order][index] == 0:
                continue
            elif nested_sums[read_order][index] <= threshold:
                nested_alignment[read_order][index] = (
                    read_order,
                    index,
                    nested_sums[read_order][index],
                )
            elif read_order == highest_order:
                raise ValueError(
                    f"""single pixel count {
                        nested_sums[read_order][index]} exceeds threshold {threshold}"""
                )
    return nested_alignment[highest_order]


def generate_destination_pixel_map(histogram, pixel_map):
    """Generate mapping from destination pixel to all the constituent pixels.
    Args:
        histogram (:obj:`np.array`): one-dimensional numpy array of long integers where the
            value at each index corresponds to the number of objects found at the healpix pixel.
        pixel_map (:obj:`np.array`): one-dimensional numpy array of integer 3-tuples.
            See `histogram.generate_alignment` for more details on this format.
    Returns:
        dictionary that maps the integer 3-tuple of a pixel at destination order to the set of
        indexes in histogram for the pixels at the original healpix order
    """

    non_none_elements = [i for i in pixel_map if i is not None]
    unique_pixels = np.unique(non_none_elements, axis=0)

    result = {}
    for pixel in unique_pixels:
        source_pixels = []
        for i, source in enumerate(pixel_map):
            if not source:
                continue
            if source[0] == pixel[0] and source[1] == pixel[1] and histogram[i] > 0:
                source_pixels.append(i)
        result[tuple(pixel)] = source_pixels

    return result
