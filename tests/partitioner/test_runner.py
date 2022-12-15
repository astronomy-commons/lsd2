"""Test selection of execution runner"""

import os
import tempfile

import data_paths as dc
import pytest

import partitioner.runner as runner
from partitioner.arguments import PartitionArguments


def test_unsupported():
    """Test using un-supported runner type"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        args = PartitionArguments()
        args.from_params(
            catalog_name="runner_test",
            input_path=dc.TEST_SMALL_SKY_PARTS_DATA_DIR,
            input_format="csv",
            output_path=tmp_dir,
        )
        # Get around argument parsing validation =]
        args.runtime = "unsupported"
        with pytest.raises(NotImplementedError):
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


def test_small_sky_single():
    """Test runner with single execution"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        args = PartitionArguments()
        args.from_params(
            catalog_name="small_sky",
            input_path=dc.TEST_SMALL_SKY_PARTS_DATA_DIR,
            input_format="csv",
            output_path=tmp_dir,
            highest_healpix_order=1,
            ra_column="ra",
            dec_column="dec",
            progress_bar=False,
        )

        runner.run(args)

        assert os.path.exists(os.path.join(args.catalog_path, "small_sky_meta.json"))
