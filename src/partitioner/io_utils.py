"""Utility functions for reading and writing files"""

import json
import os

import numpy as np
import pandas as pd
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

    dumped_metadata = json.dumps(metadata, indent=4, cls=NumpyEncoder)
    metadata_filename = os.path.join(args.output_path, f"{args.catalog_name}_meta.json")
    print(metadata_filename)
    with open(
        metadata_filename,
        "w",
        encoding="utf-8",
    ) as metadata_file:
        metadata_file.write(dumped_metadata + "\n")
