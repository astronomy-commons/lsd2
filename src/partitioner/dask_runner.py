"""Partitioner runner that uses dask for parallelization"""

import numpy as np
from dask.distributed import Client, progress, wait

import partitioner.histogram as hist
import partitioner.io_utils as io_utils
import partitioner.map_reduce as mr
from partitioner.arguments import PartitionArguments


def _map_to_pixels(
    input_file,
    file_format,
    highest_order,
    ra_column,
    dec_column,
    shard_suffix,
    cache_path,
):
    """Wrapper around map_reduce.map_to_pixels that loads the dataframe"""

    return mr.map_to_pixels(
        data=io_utils.read_dataframe(input_file, file_format),
        highest_order=highest_order,
        ra_column=ra_column,
        dec_column=dec_column,
        shard_suffix=shard_suffix,
        cache_path=cache_path,
    )


def _generate_histogram(args, client):
    """Generate a raw histogram of object counts in each healpix pixel"""

    futures = []
    for i, file_path in enumerate(args.input_paths):
        if args.debug_stats_only:
            futures.append(
                client.submit(
                    hist.generate_partial_histogram,
                    file_path,
                    highest_order=args.highest_healpix_order,
                    file_format=args.input_format,
                    ra_column=args.ra_column,
                    dec_column=args.dec_column,
                )
            )
        else:
            futures.append(
                client.submit(
                    _map_to_pixels,
                    input_file=file_path,
                    file_format=args.input_format,
                    highest_order=args.highest_healpix_order,
                    ra_column=args.ra_column,
                    dec_column=args.dec_column,
                    shard_suffix=i,
                    cache_path=args.tmp_dir,
                )
            )
    if args.progress_bar:
        progress(futures)
    else:
        wait(futures)

    raw_histogram = hist.empty_histogram(args.highest_healpix_order)
    for future in futures:
        raw_histogram = np.add(raw_histogram, future.result())
    return raw_histogram


def _reduce_pixels(args, destination_pixel_map, client):
    """Loop over destination pixels and merge into parquet files"""

    futures = []
    for _, destination_pixel in destination_pixel_map.iterrows():
        futures.append(
            client.submit(
                mr.reduce_shards,
                cache_path=args.tmp_dir,
                origin_pixel_numbers=destination_pixel["origin_pixels"],
                destination_pixel_order=destination_pixel["order"],
                destination_pixel_number=destination_pixel["pixel"],
                destination_pixel_size=destination_pixel["num_objects"],
                output_path=args.catalog_path,
                id_column=args.id_column,
            )
        )
    if args.progress_bar:
        progress(futures)
    else:
        wait(futures)


def _validate_args(args):
    if not args:
        raise ValueError("args is required and should be type PartitionArguments")
    if not isinstance(args, PartitionArguments):
        raise ValueError("args must be type PartitionArguments")
    if not args.runtime == "dask":
        raise ValueError(f'runtime mismatch ({args.runtime} should be "dask"')


def run(args):
    """Partitioner runner that creates a dask client from the arguments"""
    _validate_args(args)

    with Client(
        local_directory=args.dask_tmp,
        n_workers=args.dask_n_workers,
        threads_per_worker=args.dask_threads_per_worker,
    ) as client:
        run_with_client(args, client)


def run_with_client(args, client):
    """Partitioner runner, where the client context may out-live the runner"""
    _validate_args(args)
    raw_histogram = _generate_histogram(args, client)
    pixel_map = hist.generate_alignment(
        raw_histogram, args.highest_healpix_order, args.pixel_threshold
    )
    destination_pixel_map = hist.generate_destination_pixel_map(
        raw_histogram, pixel_map
    )
    io_utils.write_legacy_metadata(args, raw_histogram, pixel_map)
    io_utils.write_catalog_info(args, raw_histogram)
    io_utils.write_partition_info(args, destination_pixel_map)

    if not args.debug_stats_only:
        _reduce_pixels(args, destination_pixel_map, client)
