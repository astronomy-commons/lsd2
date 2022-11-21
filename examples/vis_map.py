"""Little script to render a sky map of the pixels in the partitioned catalog"""

import json

import healpy as hp
import numpy as np
from matplotlib import pyplot as plt


def plot_map(c, k, title=""):
    """
    Debugging plotter.
    """
    npix = hp.order2npix(k)
    orders = np.full(npix, hp.pixelfunc.UNSEEN)
    idx = np.arange(npix)
    c_orders = [int(x) for x in c.keys()]
    c_orders.sort()

    for o in c_orders:
        k2o = 4 ** (k - o)
        pixs = c[str(o)]
        pixk = idx.reshape(-1, k2o)[pixs].flatten()
        orders[pixk] = o

    hp.mollview(orders, max=k, title=title, nest=True)
    print("got a mollview")
    plt.show()


if __name__ == "__main__":
    hips_meta_fn = "/home/delucchi/xmatch/catalogs/output/td_demo/td_demo_meta.json"
    with open(hips_meta_fn) as f:
        catalog_md = json.load(f)

    catalog_hips = catalog_md["hips"]
    k = max([int(x) for x in catalog_hips.keys()])
    plot_map(c=catalog_hips, k=k)
