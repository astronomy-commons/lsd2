"""Utility functions for reading and writing files"""

import os

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