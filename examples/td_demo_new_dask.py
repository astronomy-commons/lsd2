"""Main method to enable command line execution"""

import tempfile

import partitioner.dask_runner as dr
from partitioner.arguments import PartitionArguments

if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp_dir:
        args = PartitionArguments()
        args.from_params(
            catalog_name="td_demo",
            input_path="/home/delucchi/td_data/wsource",
            input_format="parquet",
            ra_column="ra",
            dec_column="decl",
            id_column="diaObjectId",
            pixel_threshold=1_000_000,
            dask_tmp=tmp_dir,
            runtime="dask",
            highest_healpix_order=6,
            output_path="/home/delucchi/xmatch/catalogs/",
        )
        dr.run(args)
