import pandas as pd
import time


if __name__ == "__main__":
    s = time.time()

    pd.read_parquet(
        # "/home/delucchi/td_data/truth_tract3830.parquet", engine="pyarrow"
        "/home/delucchi/xmatch/catalogs/output/td_demo/Norder7/Npix138400/catalog.parquet", engine="pyarrow"
    ).head(2).transpose().to_csv("/home/delucchi/td_test/transpose.csv", index=True)

    e = time.time()
    print(f"Elapsed Time: {e-s}")
