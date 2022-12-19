"""Main method to enable command line execution"""

import logging
import tempfile

import partitioner.runner as sr
from partitioner.arguments import PartitionArguments

logging.basicConfig(
    filename="example.log",
    encoding="utf-8",
    level=logging.DEBUG,
    format="%(asctime)s %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
)

if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp_dir:
        args = PartitionArguments()
        args.from_params(
            catalog_name="small_sky",
            input_path="/home/delucchi/git/lsd2/tests/partitioner/data/small_sky_parts/",
            input_format="csv",
            output_path="/home/delucchi/xmatch/catalogs/",
            overwrite=True,
            highest_healpix_order=1,
            ra_column="ra",
            dec_column="dec",
        )
        sr.run(args)
