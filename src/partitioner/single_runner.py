"""Partitioner runner that doesn't use any parallelization mechanism"""

import numpy as np
from tqdm import tqdm

import partitioner.histogram as hist
import partitioner.io_utils as io_utils
import partitioner.map_reduce as mr
from partitioner.arguments import PartitionArguments


def _generate_histogram(args):
    """Generate a raw histogram of object counts in each healpix pixel"""

    raw_histogram = hist.empty_histogram(args.highest_healpix_order)
    iterator = (
        tqdm(args.input_paths, desc="Mapping ")
        if args.progress_bar
        else args.input_paths
    )
    for i, file_path in enumerate(iterator):
        if args.debug_stats_only:
            partial_histogram = hist.generate_partial_histogram(
                file_path=file_path,
                highest_order=args.highest_healpix_order,
                file_format=args.input_format,
                ra_column=args.ra_column,
                dec_column=args.dec_column,
            )
            raw_histogram = np.add(raw_histogram, partial_histogram)
        else:
            partial_histogram = mr.map_to_pixels(
                input_file=file_path,
                file_format=args.input_format,
                highest_order=args.highest_healpix_order,
                ra_column=args.ra_column,
                dec_column=args.dec_column,
                shard_index=i,
                cache_path=args.tmp_dir,
            )
            raw_histogram = np.add(raw_histogram, partial_histogram)

    return raw_histogram


def _reduce_pixels(args, destination_pixel_map):
    """Loop over destination pixels and merge into parquet files"""

    iterator = (
        tqdm(destination_pixel_map.items(), desc="Reducing")
        if args.progress_bar
        else destination_pixel_map.items()
    )
    for destination_pixel, source_pixels in iterator:
        mr.reduce_shards(
            cache_path=args.tmp_dir,
            origin_pixel_numbers=source_pixels,
            destination_pixel_order=destination_pixel[0],
            destination_pixel_number=destination_pixel[1],
            destination_pixel_size=destination_pixel[2],
            output_path=args.catalog_path,
            id_column=args.id_column,
        )


def run(args):
    """Partitioner runner"""
    if not args:
        raise ValueError("args is required and should be type PartitionArguments")
    if not isinstance(args, PartitionArguments):
        raise ValueError("args must be type PartitionArguments")
    if not args.runtime == "single":
        raise ValueError(f'runtime mismatch ({args.runtime} should be "single"')

    raw_histogram = _generate_histogram(args)
    pixel_map = hist.generate_alignment(
        raw_histogram, args.highest_healpix_order, args.pixel_threshold
    )
    io_utils.write_legacy_metadata(args, raw_histogram, pixel_map)
    io_utils.write_catalog_info(args, raw_histogram)
    io_utils.write_partition_info(args, pixel_map)

    if not args.debug_stats_only:
        destination_pixel_map = hist.generate_destination_pixel_map(
            raw_histogram, pixel_map
        )
        _reduce_pixels(args, destination_pixel_map)
