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
    for i, file_path in enumerate(
        tqdm(args.input_paths, desc="Mapping ", disable=(not args.progress_bar))
    ):
        partial_histogram = mr.map_to_pixels(
            input_file=file_path,
            file_format=args.input_format,
            filter_function=args.filter_function,
            highest_order=args.highest_healpix_order,
            ra_column=args.ra_column,
            dec_column=args.dec_column,
            shard_suffix=i,
            cache_path=None if args.debug_stats_only else args.tmp_dir,
        )
        raw_histogram = np.add(raw_histogram, partial_histogram)

    return raw_histogram


def _reduce_pixels(args, destination_pixel_map):
    """Loop over destination pixels and merge into parquet files"""

    for destination_pixel, source_pixels in tqdm(
        destination_pixel_map.items(),
        desc="Reducing",
        disable=(not args.progress_bar),
    ):
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

    progress = tqdm(total=2, desc="Binning ", disable=(not args.progress_bar))
    pixel_map = hist.generate_alignment(
        raw_histogram, args.highest_healpix_order, args.pixel_threshold
    )
    progress.update(1)
    destination_pixel_map = hist.generate_destination_pixel_map(
        raw_histogram, pixel_map
    )
    progress.update(1)
    progress.close()

    if not args.debug_stats_only:
        _reduce_pixels(args, destination_pixel_map)

    # All done - write out the metadata
    progress = tqdm(total=3, desc="Finishing", disable=(not args.progress_bar))
    io_utils.write_legacy_metadata(args, raw_histogram, pixel_map)
    progress.update(1)
    io_utils.write_catalog_info(args, raw_histogram)
    progress.update(1)
    io_utils.write_partition_info(args, destination_pixel_map)
    progress.update(1)
    progress.close()
