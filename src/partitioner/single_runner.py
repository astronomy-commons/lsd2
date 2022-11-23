"""Partitioner runner that doesn't use any parallelization mechanism"""

import numpy as np

import partitioner.histogram as hist
import partitioner.io_utils as io_utils


def _generate_histogram(args):
    """Generate a raw histogram of object counts in each healpix pixel"""

    raw_histogram = hist.empty_histogram(args.highest_healpix_order)
    for file_path in args.input_paths:

        partial_histogram = hist.generate_partial_histogram(
            file_path=file_path,
            highest_order=args.highest_healpix_order,
            file_format=args.input_format,
            ra_column=args.ra_column,
            dec_column=args.dec_column,
        )
        raw_histogram = np.add(raw_histogram, partial_histogram)

    return raw_histogram


def run(args):
    """Partitioner runner"""
    if not args:
        raise ValueError("args is required and should be type PartitionArguments")
    raw_histogram = _generate_histogram(args)
    pixel_map = hist.generate_alignment(
        raw_histogram, args.highest_healpix_order, args.pixel_threshold
    )
    io_utils.write_legacy_metadata(args, raw_histogram, pixel_map)
