import argparse
import sys
import time

import pandas as pd
import pyarrow.parquet as pq
from astropy.table import Table

if __name__ == "__main__":
    s = time.time()

    parser = argparse.ArgumentParser(
        prog="example",
        description="empty",
    )
    parser.add_argument(
        "input_file_name",
        help="path to a single csv file",
    )
    parser.add_argument(
        "output_file_name",
        help="path to a single fits file",
    )
    args = parser.parse_args(sys.argv[1:])
    input_file_name = args.input_file_name
    output_file_name = args.output_file_name

    data_frame = pd.read_csv(input_file_name)

    table = Table.from_pandas(data_frame)
    table.write(output_file_name, format="fits")

    e = time.time()
    print(f"Elapsed Time: {e-s}")
