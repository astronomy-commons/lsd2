
import dask.dataframe as dd
import pandas as pd
import numpy as np
import healpy as hp
from typing import Callable, TYPE_CHECKING, Dict

def _group_by_hc_apply(df, ufunc, **kwargs):
    return df.groupby("_hipscat_index").apply(ufunc, **kwargs)

def _skymap_agg2(df, col, ufunc=np.mean):
    assert all(df["order_pix"].values == df["order_pix"].iloc[0])
    ret_dict = {
        "pix" : [df["order_pix"].iloc[0]],
        "val" : [ufunc(df[col].values)]
    }
    return pd.DataFrame(ret_dict, columns=ret_dict.keys())


def _skymap_groupby_apply(df, **kwargs):
    return df.groupby("order_pix").apply(_skymap_agg2, **kwargs)

def _assign_order_pix(df, spatial_kws, nside):
    df['order_pix'] = hp.ang2pix(nside, df[spatial_kws[0]], df[spatial_kws[1]], lonlat=True, nest=True)
    return df

class SpatialResult:



    def __init__(self, hipscat, lsddf, healpix_dict):
        self.lsddf  = lsddf
        self.healpix_dict = healpix_dict
        self.hipscat = hipscat

    def __repr__(self):
        return self.lsddf.to_string()

    def compute(self):
        return self.lsddf.compute()
    
    def query(self, *args, **kwargs):
        _ddf = self.lsddf.query(*args, **kwargs)
        self.lsddf = _ddf
        return self
    
    def assign(self, **kwargs):
        _ddf = self.lsddf.assign(**kwargs)
        self.lsddf = _ddf
        return self

    def for_each(self, ufunc, **kwargs):
        _ddf = dd.map_partitions(_group_by_hc_apply, self.lsddf, ufunc, **kwargs)
        self.lsddf = _ddf
        return self

    def skymap(self, col, f=np.mean, k=6, spatial_kws=('ra', 'dec')):
        nside = hp.order2nside(k)

        meta1 = self.lsddf._meta
        meta1['order_pix'] = 'f8'
        _ddf = self.lsddf.map_partitions(_assign_order_pix, spatial_kws=spatial_kws, nside=nside, meta=meta1)
        
        meta = {"pix": "i8", "val": "f8"}
        tf = _ddf.map_partitions(_skymap_groupby_apply, col=col, ufunc=f, meta=meta).compute()

        npix = hp.order2npix(k)
        img = np.zeros(npix)
        img[tf["pix"].values] = tf["val"].values
        return img

    def cross_match(self, othercat, **kwargs):
        return self.hipscat.cross_match(othercat=othercat, **kwargs, spatial_filter=True)
    
    def cone_search(self, **kwargs):
        return self.hipscat.cone_search(**kwargs)