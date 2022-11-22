"""Utilities for generating and manipulating object count histograms"""
from functools import partial
from operator import add

import dask
import dask.bag as db
import dask.dataframe as dd
import healpy as hp
import numpy as np
import pandas as pd
from astropy.table import Table
from dask.delayed import delayed
from dask.distributed import Client, progress

from . import io_utils


def _generate_partial_histogram(
    file_path,
    highest_order,
    file_format,
    ra_column,
    dec_column,
):
    histo = np.zeros(hp.order2npix(highest_order), dtype=np.float32)

    data = io_utils.read_dataframe(file_path, file_format)

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
    histo[mapped_pixel] += count_at_pixel.astype(np.float32)
    return histo


def generate_histogram(args=None, client=None):
    """Generate a raw histogram of object counts in each healpix pixel"""
    if not args:
        raise ValueError("args is required and should be type PartitionArguments")

    raw_histogram = np.zeros(
        hp.order2npix(args.highest_healpix_order), dtype=np.float32
    )
    for file_path in args.input_paths:

        partial_histogram = _generate_partial_histogram(
            file_path=file_path,
            highest_order=args.highest_healpix_order,
            file_format=args.input_format,
            ra_column=args.ra_column,
            dec_column=args.dec_column,
        )
        raw_histogram = np.add(raw_histogram, partial_histogram)

    return raw_histogram
