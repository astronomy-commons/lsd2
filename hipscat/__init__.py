#!/usr/bin/env python
# coding: utf-8


import dask.utils as du
try:
    _orig_natural_sort_key
except NameError:
    _orig_natural_sort_key = du.natural_sort_key

def hips_or_natural_sort_key(s): #: str) -> list[str | int]:
    # Let's reuse multi-file parquet reading routines from Dask. The only
    # obstacle is that Dask will sort them by name, which will result in
    # out-of-order partitions (the index won't be monotonically increasing)
    # As a quick-and-dirty fix, we'll monkey-patch the sort function to
    # recognize hipscat structure.
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

