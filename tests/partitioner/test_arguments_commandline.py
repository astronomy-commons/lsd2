"""Tests of command line argument validation"""

import data_paths as dc
import pytest

from partitioner.arguments import PartitionArguments


def test_none():
    """No arguments provided. Should error for required args."""
    empty_args = []
    args = PartitionArguments()
    with pytest.raises(ValueError):
        args.from_command_line(empty_args)


def test_invalid_arguments():
    """Arguments are ill-formed."""
    bad_form_args = ["catalog", "path", "path"]
    args = PartitionArguments()
    with pytest.raises(SystemExit):
        args.from_command_line(bad_form_args)


def test_invalid_path():
    """Required arguments are provided, but paths aren't found."""
    bad_path_args = ["-c", "catalog", "-i", "path", "-o", "path"]
    args = PartitionArguments()
    with pytest.raises(ValueError):
        args.from_command_line(bad_path_args)


def test_good_paths():
    """Required arguments are provided, and paths are found."""
    good_args = [
        "--catalog_name",
        "catalog",
        "--input_path",
        dc.TEST_BLANK_DATA_DIR,
        "--output_path",
        dc.TEST_TMP_DIR,
        "--input_format",
        "csv",
    ]
    args = PartitionArguments()
    args.from_command_line(good_args)
    assert args.input_path == dc.TEST_BLANK_DATA_DIR
    assert args.output_path == dc.TEST_TMP_DIR


def test_good_paths_short_names():
    """Required arguments are provided, using short names for arguments."""
    good_args = [
        "-c",
        "catalog",
        "-i",
        dc.TEST_BLANK_DATA_DIR,
        "-o",
        dc.TEST_TMP_DIR,
        "-fmt",
        "csv",
    ]
    args = PartitionArguments()
    args.from_command_line(good_args)
    assert args.input_path == dc.TEST_BLANK_DATA_DIR
    assert args.output_path == dc.TEST_TMP_DIR