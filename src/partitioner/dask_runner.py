"""Partitioner runner that doesn't use any parallelization mechanism"""

import numpy as np
from dask.distributed import Client, progress

import partitioner.histogram as hist


def _generate_histogram(args, client):
    """Generate a raw histogram of object counts in each healpix pixel"""

    futures = client.map(
        hist.generate_partial_histogram,
        args.input_paths,
        highest_order=args.highest_healpix_order,
        file_format=args.input_format,
        ra_column=args.ra_column,
        dec_column=args.dec_column,
    )
    progress(futures)

    raw_histogram = hist.empty_histogram(args.highest_healpix_order)
    for future in futures:
        raw_histogram = np.add(raw_histogram, future.result())
    return raw_histogram


def run(args):
    """Partitioner runner"""
    if not args:
        raise ValueError("args is required and should be type PartitionArguments")
    client = Client(
        local_directory=args.dask_tmp,
        n_workers=1,
        threads_per_worker=1,
    )
    _generate_histogram(args, client)
