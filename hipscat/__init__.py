#!/usr/bin/env python
# coding: utf-8

# Let's reuse multi-file parquet reading routines from Dask. The only
# obstacle is that Dask will sort them by name, which will result in
# out-of-order partitions (the index won't be monotonically increasing)
# As a quick-and-dirty fix, we'll monkey-patch the sort function to
# recognize hipscat structure.

import dask.utils as du
try:
    _orig_natural_sort_key
except NameError:
    _orig_natural_sort_key = du.natural_sort_key

def hips_or_natural_sort_key(s): #: str) -> list[str | int]:
    import re
    m = re.match(r"^(.*)/Norder(\d+)/Npix(\d+)/([^/]*)$", s)
    if m is None:
        return _orig_natural_sort_key(s)
    
    root, order, ipix, leaf = m.groups()
    order, ipix = int(order), int(ipix)
    ipix20 = ipix << 2*(20 - order)
    k = (root, ipix20, leaf)
    return k
hips_or_natural_sort_key.__doc__ = _orig_natural_sort_key.__doc__

du.natural_sort_key = hips_or_natural_sort_key

import numpy as np
import healpy as hp
import pandas as pd
import os
import os.path
from tqdm import tqdm

from . import catalog
from . import partitioner
from . import util

Catalog = catalog.Catalog
Partitioner = partitioner.Partitioner

def import_sample_data(location=os.getcwd(), client=None):
    gaia_id_kw = 'source_id'
    gaia_ra_kw = 'ra'
    gaia_dec_kw = 'dec'
    file_source = 'http://cdn.gea.esac.esa.int/Gaia/gdr3/gaia_source/'
    sample_threshold = 500_000
    fmt = 'csv.gz'
    skiprows = np.arange(0,1000)

    urls1 = util.get_csv_urls(url=file_source, fmt=fmt)[0:10]
    partitioner1 = Partitioner(catname='gaia_exA', fmt=fmt, urls=urls1, id_kw=gaia_id_kw,
                    order_k=10, verbose=True, debug=False, ra_kw=gaia_ra_kw, dec_kw=gaia_dec_kw,
                    threshold=sample_threshold, location=location, skiprows=skiprows)
    partitioner1.run(client=client)

    urls2 = util.get_csv_urls(url=file_source, fmt=fmt)[5:15]
    partitioner2 = Partitioner(catname='gaia_exB', fmt=fmt, urls=urls2, id_kw=gaia_id_kw,
                    order_k=10, verbose=True, debug=False, ra_kw=gaia_ra_kw, dec_kw=gaia_dec_kw,
                    threshold=sample_threshold, location=location, skiprows=skiprows)
    partitioner2.run(client=client)

def main():
    # HACK: for quick tests, run python -m hipscat
    #    import glob
    #    urls = glob.glob('./gdr2/*.csv.gz')
    from dask.distributed import Client
    client = Client()
    import_sample_data(client=client)
