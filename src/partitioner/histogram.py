"""Utilities for generating and manipulating object count histograms"""
from functools import partial

import dask.bag as db


def _generate_partial_histogram():
    # TODO
    histo = []
    return histo


def generate_histogram(args=None):
    """Generate a raw histogram of object counts in each healpix pixel"""
    if not args:
        raise ValueError("args is required and should be type PartitionArguments")
    raw_histogram = (
        db.from_sequence(args.input_paths, partition_size=1)
        .reduction(
            partial(
                _generate_partial_histogram,
                highest_order=args.highest_order,
                ra_column=args.ra_column,
                dec_column=args.dec_column,
            ),
            sum,
            split_every=3,
        )
        .compute()
    )
    return raw_histogram
