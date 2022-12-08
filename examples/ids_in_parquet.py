import argparse
import sys
import time

import pandas as pd
import pyarrow.parquet as pq

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
    parser.add_argument("id_column", default="diaObjectId")
    args = parser.parse_args(sys.argv[1:])
    file_name = args.file_name
    id_column = args.id_column

    parquet_file = pq.ParquetFile(file_name)

    data_frame = pd.read_parquet(file_name, engine="pyarrow")

    assert id_column in data_frame.columns
    ids = data_frame[id_column].tolist()
    print(len(ids))
    set_ids = [*set(ids)]
    print(len(set_ids))

    e = time.time()
    print(f"Elapsed Time: {e-s}")
