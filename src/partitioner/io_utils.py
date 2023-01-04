"""Utility functions for reading and writing files"""

import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import pkg_resources
import pyarrow as pa
import pyarrow.parquet as pq
from astropy.table import Table


def read_dataframe(path="", file_format="parquet") -> pd.DataFrame:
    """Read a file in as a dataframe.

    Currently supported file formats include:
        - `csv` - comma separated values
        - `parquet` - apache columnar data format
        - `fits` - flexible image transport system

    Args:
        path (str): fully-specified path to the file
        file_format (str): expected file format for the input file. This
            likely matches the extension of the file, but doesn't need to.
    Returns:
        dataframe with the data content at the target file
    Raises:
        FileNotFoundError: if there is no regular file at the indicated `path`.
        NotImplementedError: if the file format is not yet supported.
    """

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
    elif file_format == "parquet":
        data_frame = pd.read_parquet(path, engine="pyarrow")
    elif file_format == "fits":
        dat = Table.read(path, format="fits")
        data_frame = dat.to_pandas()
    else:
        raise NotImplementedError(f"File Format: {file_format} not supported")

    return data_frame


class NumpyEncoder(json.JSONEncoder):
    """Special json encoder for numpy integer types"""

    def default(self, o):
        int_object = o
        if isinstance(int_object, (np.int64, np.ulonglong)):
            return int(int_object)


def write_json_file(metadata_dictionary, file_name):
    """Convert metadata_dictionary to a json string and print to file.
    Args:
        metadata_dictionary (:obj:`dictionary`): a dictionary of key-value pairs
        file_name (str): destination for the json file
    """
    dumped_metadata = json.dumps(metadata_dictionary, indent=4, cls=NumpyEncoder)
    with open(
        file_name,
        "w",
        encoding="utf-8",
    ) as metadata_file:
        metadata_file.write(dumped_metadata + "\n")

def write_catalog_info(args, histogram):
    """Write a catalog_info.json file with catalog metadata

    Args:
        args (:obj:`PartitionArguments`): collection of runtime arguments for the partitioning job
        histogram (:obj:`np.array`): one-dimensional numpy array of long integers where the
            value at each index corresponds to the number of objects found at the healpix pixel.
    """
    metadata = {}
    metadata["cat_name"] = args.catalog_name
    metadata["version"] = pkg_resources.get_distribution("lsd2").version
    now = datetime.now()
    metadata["generation_date"] = now.strftime("%Y.%m.%d")
    metadata["ra_kw"] = args.ra_column
    metadata["dec_kw"] = args.dec_column
    metadata["id_kw"] = args.id_column
    metadata["total_objects"] = histogram.sum()

    metadata["pixel_threshold"] = args.pixel_threshold

    metadata_filename = os.path.join(args.catalog_path, "catalog_info.json")
    write_json_file(metadata, metadata_filename)


def write_partition_info(args, pixel_map):
    """Write all partition data to CSV file.
    Args:
        args (:obj:`PartitionArguments`): collection of runtime arguments for the partitioning job
        pixel_map (:obj:`np.array`): one-dimensional numpy array of integer 3-tuples.
            See `histogram.generate_alignment` for more details on this format.
    """
    metadata_filename = os.path.join(args.catalog_path, "partition_info.csv")
    temp = [i for i in pixel_map if i is not None]
    partitions = np.unique(temp, axis=0)
    data_frame = pd.DataFrame(partitions)
    data_frame.columns = ["order", "pixel", "num_objects"]
    data_frame = data_frame.astype(int)
    data_frame.to_csv(metadata_filename, index=False)

def write_legacy_metadata(args, histogram, pixel_map):
    """Write a <catalog_name>_meta.json with the format expected by the prototype catalog implementation

    Args:
        args (:obj:`PartitionArguments`): collection of runtime arguments for the partitioning job
        histogram (:obj:`np.array`): one-dimensional numpy array of long integers where the
            value at each index corresponds to the number of objects found at the healpix pixel.
        pixel_map (:obj:`np.array`): one-dimensional numpy array of integer 3-tuples.
            See `histogram.generate_alignment` for more details on this format.
    """
    metadata = {}
    metadata["cat_name"] = args.catalog_name
    metadata["ra_kw"] = args.ra_column
    metadata["dec_kw"] = args.dec_column
    metadata["id_kw"] = args.id_column
    metadata["n_sources"] = histogram.sum()
    metadata["pix_threshold"] = args.pixel_threshold
    metadata["urls"] = args.input_paths

    hips_structure = {}
    temp = [i for i in pixel_map if i is not None]
    unique_partitions = np.unique(temp, axis=0)
    for item in unique_partitions:
        order = int(item[0])
        pixel = int(item[1])
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
        tables.append(pq.read_table(path))
    merged_table = pa.concat_tables(tables)
    if sorting:
        merged_table = merged_table.sort_by(sorting)

    pq.write_table(merged_table, where=output_file_name)

    return len(merged_table)
