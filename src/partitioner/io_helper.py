"""Utility functions for reading and writing files"""

import pandas as pd
from astropy.table import Table


def read_dataframe(path="", file_format="parquet"):
    """Read a file in as a dataframe"""

    # load the input file
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
