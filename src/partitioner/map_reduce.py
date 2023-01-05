"""Methods for performing the partitioning map reduce operation"""

import os

import healpy as hp
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from astropy.table import Table

from partitioner.histogram import empty_histogram
from partitioner.io_utils import concatenate_parquet_files


def _map_dataframe(
    data,
    highest_order,
    ra_column,
    dec_column,
    shard_suffix,
    cache_path=None,
    filter_function=None,
):
    """Inner method to perform per-dataframe mapping"""

    histo = empty_histogram(highest_order)
    if filter_function:
        data = filter_function(data)
    data.reset_index(inplace=True)

    mapped_pixels = hp.ang2pix(
        2**highest_order,
        data[ra_column].values,
        data[dec_column].values,
        lonlat=True,
        nest=True,
    )
    mapped_pixel, count_at_pixel = np.unique(mapped_pixels, return_counts=True)
    histo[mapped_pixel] += count_at_pixel.astype(np.ulonglong)

    if cache_path:
        for pixel in mapped_pixel:
            data_indexes = np.where(mapped_pixels == pixel)
            filtered_data = data.filter(items=data_indexes[0].tolist(), axis=0)

            pixel_dir = os.path.join(cache_path, f"pixel_{pixel}")
            os.makedirs(pixel_dir, exist_ok=True)
            output_file = os.path.join(pixel_dir, f"shard_{shard_suffix}.parquet")
            filtered_data.to_parquet(output_file)
    return histo


def map_to_pixels(
    input_file,
    file_format,
    highest_order,
    ra_column,
    dec_column,
    shard_suffix,
    cache_path=None,
    filter_function=None,
):
    """Map a file if input objects to their healpix pixels.

    Copy objects into pixel-specific parquet files and return histogram
    of counts for objects found in the indicated input_file

    Args:
        input_file (str): fully-specified path to an input file
        file_format (str): expected format for the input file. See io_utils.read_dataframe
            for accepted formats.
        highest_order (int):  the highest healpix order (e.g. 0-10)
        ra_column (str): where in the input to find the celestial coordinate, right ascension
        dec_column (str): where in the input to find the celestial coordinate, declination
        shard_suffix (int): unique string for this shard of mapped data.
            if mapping from multiple input files, this can be the index
            in the list of input files.
        cache_path (str): directory where temporary pixel parquet files
            will be written. If not provided, we don't write partitioned files.

    Returns:
        one-dimensional numpy array of long integers where the value at each index corresponds
        to the number of objects found at the healpix pixel.
    Raises:
        ValueError: if the `ra_column` or `dec_column` cannot be found in the input file.
        FileNotFoundError: See io_utils.read_dataframe for other error
    """

    histo = empty_histogram(highest_order)

    # Perform checks on the provided path
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"File not found at path: {input_file}")
    if not os.path.isfile(input_file):
        raise FileNotFoundError(
            f"Directory found at path - requires regular file: {input_file}"
        )

    required_columns = [ra_column, dec_column]
    # Load file using appropriate mechanism
    if "csv" in file_format:
        data = pd.read_csv(input_file)
        if not all(x in data.columns for x in required_columns):
            raise ValueError(
                f"Invalid column names in input file: {ra_column}, {dec_column} not in {input_file}"
            )
        histo = _map_dataframe(
            data,
            highest_order=highest_order,
            ra_column=ra_column,
            dec_column=dec_column,
            shard_suffix=shard_suffix,
            cache_path=cache_path,
            filter_function=filter_function,
        )
    elif file_format == "fits":
        dat = Table.read(input_file, format="fits")
        data = dat.to_pandas()
        if not all(x in data.columns for x in required_columns):
            raise ValueError(
                f"Invalid column names in input file: {ra_column}, {dec_column} not in {input_file}"
            )
        histo = _map_dataframe(
            data,
            highest_order=highest_order,
            ra_column=ra_column,
            dec_column=dec_column,
            shard_suffix=shard_suffix,
            cache_path=cache_path,
            filter_function=filter_function,
        )
    elif file_format == "parquet":
        # Chunk parquet files into more manageable sections

        full_parquet_table = pq.read_table(input_file)

        for j, smaller_table in enumerate(
            full_parquet_table.to_batches(max_chunksize=100_000)
        ):
            batch_suffix = f"{shard_suffix}_{j}"
            data = pa.Table.from_batches([smaller_table]).to_pandas()

            if not all(x in data.columns for x in required_columns):
                raise ValueError(
                    f"Invalid column names in input file: {ra_column}, {dec_column} not in {input_file}"
                )
            histo = np.add(
                histo,
                _map_dataframe(
                    data,
                    highest_order=highest_order,
                    ra_column=ra_column,
                    dec_column=dec_column,
                    shard_suffix=batch_suffix,
                    cache_path=cache_path,
                    filter_function=filter_function,
                ),
            )
    else:
        raise NotImplementedError(f"File Format: {file_format} not supported")

    return histo


def reduce_shards(
    cache_path,
    origin_pixel_numbers,
    destination_pixel_order,
    destination_pixel_number,
    destination_pixel_size,
    output_path,
    id_column,
):
    """Reduce sharded source pixels into destination pixels.

    Args:
        cache_path (str): directory where temporary pixel parquet files
            were written, and will be read from in this step
        origin_pixel_numbers (:obj:`int[]`) list of pixel numbers at original healpix order
        destination_pixel_order (int): the healpix order of the destination pixel
        destination_pixel_number (int): the destination healpix pixel
        destination_pixel_size (int): the number of objects expected to be in the destination pixel
        output_path (str): directory/prefix where destination parquet files will be written
        id_column (str): the id column or other column used for sorting the output
    Raises:
        ValueError: if the number of objects in the destination parquet file does not
            match the provided `destination_pixel_size`.
    """
    destination_dir = os.path.join(
        output_path,
        f"Norder{int(destination_pixel_order)}/Npix{int(destination_pixel_number)}",
    )
    os.makedirs(destination_dir, exist_ok=True)

    destination_file = os.path.join(destination_dir, "catalog.parquet")

    input_directories = []

    for pixel in origin_pixel_numbers:
        pixel_dir = os.path.join(cache_path, f"pixel_{pixel}")
        input_directories.append(pixel_dir)

    rows_written = concatenate_parquet_files(
        input_directories=input_directories,
        output_file_name=destination_file,
        sorting=id_column,
    )

    if rows_written != destination_pixel_size:
        raise ValueError(
            "Unexpected number of objects at pixel "
            f"({destination_pixel_order}, {destination_pixel_number})."
            f" Expected {destination_pixel_size}, wrote {rows_written}"
        )
