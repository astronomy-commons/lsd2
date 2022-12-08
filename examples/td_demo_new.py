"""Main method to enable command line execution"""

import partitioner.single_runner as sr
from partitioner.arguments import PartitionArguments

if __name__ == "__main__":
    args = PartitionArguments()
    args.from_params(
        catalog_name="td_demo",
        input_path="/home/delucchi/td_data/wsource",
        input_format="parquet",
        ra_column="ra",
        dec_column="decl",
        id_column="diaObjectId",
        pixel_threshold=1_000_000,
        highest_healpix_order=6,
        output_path="/home/delucchi/xmatch/catalogs/",
    )
    sr.run(args)
