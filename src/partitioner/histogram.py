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


def generate_alignment(histogram, highest_order=10, threshold=1_000_000):
    """Generate alignment from high order pixels to those of equal or lower order"""

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
            # print(f"{read_order} : {index} : {parent_order} : {parent_pixel}")
            nested_sums[parent_order][parent_pixel] += nested_sums[read_order][index]

    nested_alignment = []
    for i in range(0, highest_order + 1):
        nested_alignment.append(np.full(hp.order2npix(i), None))

    # work forward -
    for index in range(0, len(nested_sums[0])):
        if nested_sums[0][index] > 0 and nested_sums[0][index] <= threshold:
            nested_alignment[0][index] = (0, index, nested_sums[0][index])

    for read_order in range(1, highest_order + 1):
        parent_order = read_order - 1
        for index in range(0, len(nested_sums[read_order])):
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
