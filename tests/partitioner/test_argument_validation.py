"""Tests of argument validation, in the absense of command line parsing"""


import tempfile

import data_paths as dc
import pytest

from partitioner.arguments import PartitionArguments


def test_none():
    """No arguments provided. Should error for required args."""
    args = PartitionArguments()
    with pytest.raises(ValueError):
        args.from_params()


def test_invalid_path():
    """Required arguments are provided, but paths aren't found."""
    args = PartitionArguments()
    with pytest.raises(ValueError):
        args.from_params(catalog_name="catalog", input_path="path", output_path="path")


def test_good_paths():
    """Required arguments are provided, and paths are found."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        args = PartitionArguments()
        args.from_params(
            catalog_name="catalog",
            input_path=dc.TEST_BLANK_DATA_DIR,
            input_format="csv",
            output_path=tmp_dir,
        )
        assert args.input_path == dc.TEST_BLANK_DATA_DIR
        assert len(args.input_paths) == 1
        assert args.input_paths[0] == dc.TEST_BLANK_CSV
        assert args.output_path == tmp_dir


def test_multiple_files_in_path():
    """Required arguments are provided, and paths are found."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        args = PartitionArguments()
        args.from_params(
            catalog_name="catalog",
            input_path=dc.TEST_SMALL_SKY_PARTS_DATA_DIR,
            input_format="csv",
            output_path=tmp_dir,
        )
        assert args.input_path == dc.TEST_SMALL_SKY_PARTS_DATA_DIR
        assert len(args.input_paths) == 5


def test_single_debug_file():
    """Required arguments are provided, and paths are found."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        args = PartitionArguments()
        args.from_params(
            catalog_name="catalog",
            debug_input_files=dc.TEST_FORMATS_HEADERS_CSV,
            input_format="csv",
            output_path=tmp_dir,
        )
        assert len(args.input_paths) == 1
        assert args.input_paths[0] == dc.TEST_FORMATS_HEADERS_CSV
