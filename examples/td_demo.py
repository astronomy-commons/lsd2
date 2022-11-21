"""Script to generate partitioned catalog using older version of partitioner"""

import sys
import time

import hipscat as hc
from dask.distributed import Client

if __name__ == "__main__":
    s = time.time()
    client = Client(
        local_directory="/home/delucchi/dask/tmp/",
        n_workers=1,
        threads_per_worker=1,
    )

    c = hc.Catalog("td_demo", location="/home/delucchi/xmatch/catalogs/")
    c.hips_import(
        file_source="/home/delucchi/td_data/",
        fmt="parquet",
        ra_kw="ra",
        dec_kw="dec",
        id_kw="id",
        debug=False,
        verbose=True,
        threshold=1_000_000,
        client=client,
    )

    e = time.time()
    print(f"Elapsed Time: {e-s}")
