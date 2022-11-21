import pandas as pd
import time
import argparse
import pyarrow.parquet as pq
import sys


if __name__ == "__main__":
    s = time.time()

    parser = argparse.ArgumentParser(
        prog="LSD2 Partitioner",
        description="Instantiate a partitioned catalog from unpartitioned sources",
    )
    parser.add_argument(
        "file_name",
        help="path to a single parquet file",
    )
    args = parser.parse_args(sys.argv[1:])
    file_name = args.file_name

    parquet_file = pq.ParquetFile(file_name)
    print(parquet_file.metadata)
    print(parquet_file.schema)
    print(parquet_file.metadata.row_group(0).column(0))

    pd.read_parquet(file_name, engine="pyarrow").head(2).transpose().to_csv(
        "/home/delucchi/td_test/transpose.csv", index=True
    )

    e = time.time()
    print(f"Elapsed Time: {e-s}")
