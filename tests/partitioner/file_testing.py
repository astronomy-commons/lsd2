"""Set of convenience methods for testing file contents"""

import os
import re

import numpy.testing as npt
import pandas as pd


def assert_text_file_matches(expected_lines, file_name):
    """
    Convenience method to read a text file and compare the contents,
    line for line, against regular expressions.

    It can be easier to see differences in indivudual lines
    when file contents grow to be large.
    """
    assert os.path.exists(file_name), f"file not found [{file_name}]"
    metadata_file = open(
        file_name,
        "r",
        encoding="utf-8",
    )

    contents = metadata_file.readlines()

    assert len(expected_lines) == len(contents)
    for i, expected in enumerate(expected_lines):
        assert re.match(expected, contents[i])

    metadata_file.close()


def assert_parquet_file_ids(file_name, id_column, expected_ids):
    """
    Convenience method to read a parquet file and compare
    the object IDs to a list of expected objects.

    It can be easier to see differences in indivudual lines
    when file contents grow to be large.
    """
    assert os.path.exists(file_name), f"file not found [{file_name}]"

    data_frame = pd.read_parquet(file_name, engine="pyarrow")
    assert id_column in data_frame.columns
    ids = data_frame[id_column].tolist()
    print(ids)

    assert len(ids) == len(
        expected_ids
    ), f"object list not the same size ({len(ids)} vs {len(expected_ids)})"

    npt.assert_array_equal(ids, expected_ids)
