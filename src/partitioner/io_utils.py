"""Utility functions for reading and writing files"""

import json
import os

import numpy as np
import pandas as pd
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
        if isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


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
    for item in pixel_map:
        if not item:
            continue
        order = item[0]
        pixel = item[1]
        if order not in hips_structure:
            hips_structure[order] = []
        hips_structure[order].append(pixel)

    metadata["hips"] = hips_structure

    metadata_filename = os.path.join(args.output_path, f"{args.catalog_name}_meta.json")
    write_json_file(metadata, metadata_filename)
