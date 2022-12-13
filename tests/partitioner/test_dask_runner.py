"""Test selection of execution runner"""

import tempfile

import data_paths as dc
import pytest

import partitioner.dask_runner as runner
from partitioner.arguments import PartitionArguments


def test_unsupported():
    """Test using un-supported runner type"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        args = PartitionArguments()
        args.from_params(
            catalog_name="runner_test",
            input_path=dc.TEST_SMALL_SKY_PARTS_DATA_DIR,
            input_format="csv",
            runtime="single",
            output_path=tmp_dir,
        )
        with pytest.raises(ValueError):
            runner.run(args)


def test_empty_args():
    """Runner should fail with empty arguments"""
    with pytest.raises(ValueError):
        runner.run(None)


def test_bad_args():
    """Runner should fail with mis-typed arguments"""
    args = {"runtime": "single"}
    with pytest.raises(ValueError):
        runner.run(args)
