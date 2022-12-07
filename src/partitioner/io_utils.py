"""Utility functions for reading and writing files"""

import json
import os

import numpy as np
import pandas as pd
import pyarrow as pa
from astropy.table import Table


def read_dataframe(path="", file_format="parquet") -> pd.DataFrame:
    """Read a file in as a dataframe"""

    # Perform checks on the provided path
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found at path: {path}")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Directory found at path - requires regular file: {path}"
        )

    data_frame = pd.DataFrame
    # Load file using appropriate mechanism
    if "csv" in file_format:
        data_frame = pd.read_csv(path)
    elif "parquet" in file_format:
        data_frame = pd.read_parquet(path, engine="pyarrow")
    elif "fits" in file_format:
        dat = Table.read(path, format="fits")
        data_frame = dat.to_pandas()
    else:
        raise NotImplementedError(f"File Format: {file_format} not supported")

    return data_frame


class NumpyEncoder(json.JSONEncoder):
    """Special json encoder for numpy types"""

    def default(self, o):
        obj = o
        if isinstance(
            obj,
            (
                np.int_,
                np.intc,
                np.intp,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
                np.ulonglong,
            ),
        ):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def write_json_file(metadata_dictionary, file_name):
    """Convert metadata_dictionary to a json string and print to file."""
    dumped_metadata = json.dumps(metadata_dictionary, indent=4, cls=NumpyEncoder)
    with open(
        file_name,
        "w",
        encoding="utf-8",
    ) as metadata_file:
        metadata_file.write(dumped_metadata + "\n")


def write_catalog_info(args, histogram):
    """Write a catalog_info.json file with catalog metadata"""
    metadata = {}
    metadata["cat_name"] = args.catalog_name
    # TODO - versioning
    metadata["version"] = "0001"
    # TODO - date formatting
    metadata["generation_date"] = "0001"
    metadata["ra_kw"] = args.ra_column
    metadata["dec_kw"] = args.dec_column
    metadata["id_kw"] = args.id_column
    metadata["total_objects"] = histogram.sum()
    # metadata["orders"] = [0, 1]  # TODO

    metadata["pixel_threshold"] = args.pixel_threshold

    metadata_filename = os.path.join(args.catalog_path, "catalog_info.json")
    write_json_file(metadata, metadata_filename)


def write_partition_info(args, pixel_map):
    """Write all partition data to CSV file."""
    metadata_filename = os.path.join(args.catalog_path, "partition_info.csv")
    temp = [i for i in pixel_map if i is not None]
    partitions = np.unique(temp, axis=0)
    data_frame = pd.DataFrame(partitions)
    data_frame.columns = ["order", "pixel", "num_objects"]
    data_frame.to_csv(metadata_filename, index=False)


def write_legacy_metadata(args, histogram, pixel_map):
    """Write a <catalog_name>_meta.json with the format expected by the legacy catalog"""
    metadata = {}
    metadata["cat_name"] = args.catalog_name
    metadata["ra_kw"] = args.ra_column
    metadata["dec_kw"] = args.dec_column
    metadata["id_kw"] = args.id_column
    metadata["n_sources"] = histogram.sum()
    metadata["pix_threshold"] = args.pixel_threshold
    metadata["urls"] = args.input_paths

    hips_structure = {}
    for item in pixel_map:
        if not item:
            continue
        order = item[0]
        pixel = item[1]
        if order not in hips_structure:
            hips_structure[order] = []
        hips_structure[order].append(pixel)

    metadata["hips"] = hips_structure

    metadata_filename = os.path.join(
        args.catalog_path, f"{args.catalog_name}_meta.json"
    )
    write_json_file(metadata, metadata_filename)


def concatenate_parquet_files(input_directories, output_file_name="", sorting=""):
    """Concatenate parquet files into a single parquet file.

    Args:
        input_directories(`obj`:str list): paths to all input files
        output_file_name (str): fully-specified path to the output file
        sorting (optional str): if specified, sort by the indicated sorting
    Returns:
        count of rows written to the `output_file`.
    """

    tables = []
    for path in input_directories:
        tables.append(pa.parquet.read_table(path))
    merged_table = pa.concat_tables(tables)
    if sorting:
        merged_table = merged_table.sort_by(sorting)

    pa.parquet.write_table(merged_table, where=output_file_name)

    print(output_file_name)

    return len(merged_table)
