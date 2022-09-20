import os
import sys
import shutil
import glob
import random
import numpy as np
import pandas as pd
import healpy as hp
import matplotlib.pyplot as plt
import dask.bag as db
import dask.dataframe as dd
import dask

from astropy.table import Table
from functools import partial
from dask.distributed import Client, progress
from dask.delayed import delayed


def _gather_statistics_hpix_hist(parts, k, cache_dir, fmt, ra_kw, dec_kw):
    # histogram the list of parts, and return it
    img = np.zeros(hp.order2npix(k), dtype=np.float32)
    for fn in parts:
        parqFn = os.path.join(cache_dir, os.path.basename(fn).split('.')[0] + '.parquet')
        if not os.path.exists(parqFn):
            # load the input file
            if 'csv' in fmt:
                df = pd.read_csv(fn)
            elif 'parquet' in fmt:
                df = pd.read_parquet(fn, engine='pyarrow')
            elif 'fits' in fmt:
                dat = Table.read(fn, format='fits')
                df = dat.to_pandas()
            else:
                sys.exit('File Format not implemented')
            
            # cache it to a parquet file
            df.to_parquet(parqFn)
        else:
            df = pd.read_parquet(parqFn, engine='pyarrow')

        #some files don't have headers
        #input the column_index number key
        #I'm pretty sure that this doesn't include the first
        #   object of every file though.
        if all([isinstance(x, int) for x in [ra_kw, dec_kw]]):
            ra_kw = df.keys()[ra_kw]
            dec_kw = df.keys()[dec_kw]

        # compute our part of the counts map
        hpix = hp.ang2pix(2**k, df[ra_kw].values, df[dec_kw].values, lonlat=True, nest=True)
        hpix, ct = np.unique(hpix, return_counts=True)
        img[hpix] += ct.astype(np.float32)

    return img


def _write_partition_structure(url, cache_dir, output_dir, orders, opix, ra_kw, dec_kw, id_kw):

    base_filename = os.path.basename(url).split('.')[0]
    parqFn = os.path.join(cache_dir, base_filename + '.parquet')
    df = pd.read_parquet(parqFn, engine='pyarrow')

    #hack if there isn't a header
    if all([isinstance(x, int) for x in [ra_kw, dec_kw, id_kw]]):
        ra_kw = df.keys()[ra_kw]
        dec_kw = df.keys()[dec_kw]
        id_kw = df.keys()[id_kw]

    for k in orders:
        df['hips_k'] = k
        df['hips_pix'] = hp.ang2pix(2**k, df[ra_kw].values, df[dec_kw].values, lonlat=True, nest=True)

        order_df = df.loc[df['hips_pix'].isin(opix[k])]
        
        #audit_counts[k].append(len(order_df))

        #reset the df so that it doesn't include the already partitioned sources
        # ~df['column_name'].isin(list) -> sources not in order_df sources
        df = df.loc[~df[id_kw].isin(order_df[id_kw])]

        #groups the sources in order_k pixels, then outputs them to the base_filename sources
        ret = order_df.groupby(['hips_k', 'hips_pix']).apply(_to_hips, hipsPath=output_dir, base_filename=base_filename)
        
        del order_df
        if len(df) == 0:
            break

    return 0


def _map_reduce(output_dir):
    #print(output_dirs)
    #for output_dir in output_dirs:

    dfs = []
    files = os.listdir(os.path.join(output_dir))    
    if len(files) == 1:
        fn = os.path.join(output_dir, files[0])
        df = pd.read_parquet(fn, engine='pyarrow')
        new_fn = os.path.join(output_dir, 'catalog.parquet')
        os.rename(fn, new_fn)
        #shutil.copyfile(fn, new_fn)
    else:
        for f in files:
            fn = os.path.join(output_dir, f)
            dfs.append(pd.read_parquet(fn, engine='pyarrow'))
            os.remove(fn)

        df = pd.concat(dfs, sort=False)
        output_fn = os.path.join(output_dir, 'catalog.parquet')
        df.to_parquet(output_fn)

    del df
    return 0


def _to_hips(df, hipsPath, base_filename):
    # WARNING: only to be used from df2hips(); it's defined out here just for debugging
    # convenience.

    # grab the order and pix number for this dataframe. Since this function
    # is intented to be called with apply() after running groupby on (k, pix), these must
    # be the same throughout the entire dataframe
    output_parquet = True
    k, pix = df['hips_k'].iloc[0],  df['hips_pix'].iloc[0]
    assert (df['hips_k']   ==   k).all()
    assert (df['hips_pix'] == pix).all()

    # construct the output directory and filename
    dir = os.path.join(hipsPath, f'Norder{k}/Npix{pix}')
    if output_parquet:
        fn  = os.path.join(dir, f'{base_filename}_catalog.parquet')
    else:
        fn  = os.path.join(dir, f'{base_filename}_catalog.csv')

    # create dir if it doesn't exist
    os.makedirs(dir, exist_ok=True)

    # write to the file (append if it already exists)
    # also, write the header only if the file doesn't already exist
    if output_parquet:
        df.to_parquet(fn)
    else:
        df.to_csv(fn, mode='a', index=False, header=not os.path.exists(fn))

    # return the number of records written
    return len(df)


if __name__ == '__main__':
    import time
    s = time.time()
    client = Client(n_workers=12, threads_per_worker=1)
    ###run logic
    td = '/astro/users/sdwyatt/git-clones/HIPS/tests/output/gaia'
    orders = os.listdir(td)

    #results = []
    #for k in orders:
    #    parts =os.listdir(os.path.join(td, k))

    #    y = delayed(_map_reduce)(
    #        parts=parts, output_dir=td, k=k
    #    )
    #    results.append(y)

    #results = dask.compute(*results)

    dds = []
    for k in orders:
        npixs = os.listdir(os.path.join(td, k))
        for pix in npixs:
            newd = os.path.join(td, k , pix)
            test_catalog = os.path.join(newd, 'catalog.parquet')
            if os.path.exists(test_catalog):
                print(f'removing {test_catalog}')
                os.remove(test_catalog)
            #y = delayed(_map_reduce2)()
            dds.append(newd)

    futures = client.map(_map_reduce2, dds)
    progress(futures)
    #dda = dask.array.from_array(dds)
    #dask.array.map_overlap(_map_reduce2, dda, depth=1, boundary='none').compute()

    #bb = db.from_sequence(dds,partition_size=4).reduction(partial(_map_reduce2), sum, split_every=3)
    #bb.compute()

    #y = delayed(_map_reduce2)(
    #    pix=pix, output_dir=newd
    #)
    
    #print(len(results))
    #results = dask.compute(*results)


    #reduction( 
    #            partial(
    #                du._gather_statistics_hpix_hist, 
    #                    k=self.order_k, cache_dir=self.cache_dir, fmt=self.fmt,
    #                    ra_kw=self.ra_kw, dec_kw=self.dec_kw
    #                ), 
    #            sum, split_every=3

    #tt = [delayed(_map_reduce)(parts=os.listdir(os.path.join(td, k)), output_dir=td, k=k) for k in orders]

    #d = dd.from_delayed(tt)
    ###end logic
    e = time.time()
    print('Elapsed time = {}'.format(e-s))