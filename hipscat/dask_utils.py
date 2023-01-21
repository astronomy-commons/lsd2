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
from sklearn.neighbors import KDTree

try:
    from . import util
except ImportError:
    import util

def _gather_statistics_hpix_hist(parts, k, cache_dir, fmt, ra_kw, dec_kw, skiprows=None):
    # histogram the list of parts, and return it
    img = np.zeros(hp.order2npix(k), dtype=np.float32)
    for fn in parts:
        parqFn = os.path.join(cache_dir, os.path.basename(fn).split('.')[0] + '.parquet')
        if not os.path.exists(parqFn):
            # load the input file
            if 'csv' in fmt:
                if skiprows is not None and isinstance(skiprows, (list, np.ndarray)):
                    df = pd.read_csv(fn, skiprows=skiprows)
                else:
                    df = pd.read_csv(fn)
            elif 'parquet' in fmt:
                df = pd.read_parquet(fn, engine='pyarrow')
            elif 'fits' in fmt:
                dat = Table.read(fn, format='fits')
                df = dat.to_pandas()
            else:
                sys.exit(f'File Format: {fmt} not implemented! \
                Supported formats are currently: csv, csv.gz, parquet, and fits')

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

        # test if the ra, dec keywords are in the file columns
        assert all([x in df.columns for x in [ra_kw, dec_kw]]), f'Invalid spatial keywords in catalog file. {ra_kw}, {dec_kw} not in {fn}'

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


def _map_reduce(output_dir, ra_kw, dec_kw):
    #print(output_dirs)
    #for output_dir in output_dirs:

    dfs = []
    files = os.listdir(os.path.join(output_dir))

    #so it doesn't re-concatenate the original catalog, if partition isn't re-ran
    files = [x for x in files if x != 'catalog.parquet']
    if len(files) == 1:
        fn = os.path.join(output_dir, files[0])
        df = pd.read_parquet(fn, engine='pyarrow')

        df["_ID"] = util.compute_index(df[ra_kw].values, df[dec_kw].values, order=14)
        df.set_index("_ID", inplace=True)
        df.sort_index(inplace=True)

        new_fn = os.path.join(output_dir, 'catalog.parquet')
        df.to_parquet(new_fn)

        os.remove(fn)
        #shutil.copyfile(fn, new_fn)
    else:
        for f in files:
            fn = os.path.join(output_dir, f)
            dfs.append(pd.read_parquet(fn, engine='pyarrow'))
            os.remove(fn)

        df = pd.concat(dfs, sort=False)
        df["_ID"] = util.compute_index(df[ra_kw].values, df[dec_kw].values, order=14)
        df.set_index("_ID", inplace=True)
        df.sort_index(inplace=True)

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


def _cross_match2(match_cats, c1_md, c2_md, c1_cols=[], c2_cols=[],  n_neighbors=1, dthresh=0.01):

    c1 = match_cats[0]
    c2 = match_cats[1]

    c1_order = int(c1.split('Norder')[1].split('/')[0])
    c1_pix = int(c1.split('Npix')[1].split('/')[0])
    c2_order = int(c2.split('Norder')[1].split('/')[0])
    c2_pix = int(c2.split('Npix')[1].split('/')[0])

    c1_df = pd.read_parquet(c1, engine='pyarrow')
    c2_df = pd.read_parquet(c2, engine='pyarrow')

    tocull1 = False
    tocull2 = False

    if c2_order > c1_order:
        order, pix = c2_order, c2_pix
        tocull1=True
    else:
        order, pix = c1_order, c1_pix
        tocull2=True

    (clon, clat) = hp.pix2ang(hp.order2nside(order), pix, nest=True, lonlat=True)

    c1_df = util.frame_cull(
        df=c1_df, df_md=c1_md,
        order=order, pix=pix,
        cols=c1_cols,
        tocull=tocull1
    )

    c2_df = util.frame_cull(
        df=c2_df, df_md=c2_md,
        order=order, pix=pix,
        cols=c2_cols,
        tocull=tocull2
    )

    ret = pd.DataFrame()
    if len(c1_df) and len(c2_df):

        xy1 = util.frame_gnomonic(c1_df, c1_md, clon, clat)
        xy2 = util.frame_gnomonic(c2_df, c2_md, clon, clat)

        tree = KDTree(xy2, leaf_size=2)
        dists, inds = tree.query(xy1, k=n_neighbors)

        outIdx = np.arange(len(c1_df)*n_neighbors)
        leftIdx = outIdx // n_neighbors
        rightIdx = inds.ravel()
        out = pd.concat(
            [
                c1_df.iloc[leftIdx].reset_index(drop=True),   # select the rows of the left table
                c2_df.iloc[rightIdx].reset_index(drop=True)  # select the rows of the right table
            ], axis=1)  # concat the two tables "horizontally" (i.e., join columns, not append rows)

        out["_DIST"] = util.gc_dist(
            out[c1_md['ra_kw']], out[c1_md['dec_kw']],
            out[c2_md['ra_kw']], out[c2_md['dec_kw']]
        )

        ret = out.loc[out['_DIST'] < dthresh]
        #out = out.loc[out['_DIST'] < dthresh]
        #ret = len(out)

        del out, dists, inds, outIdx, leftIdx, rightIdx, xy1, xy2
    del c1_df, c2_df
    return ret


def cone_search_from_daskdf(df, c_md, ra, dec, radius, columns=None):
    '''
    mapped function for calculating the number of sources
    in a disc at position (ra, dec) at radius (radius).

    inputs->
    df:      pandas.dataframe() with columns [catalog (str)]
    c_md:    the catalog metadata (json)
    ra:      right ascension of disc (int, float)
    dec:     declination of disc (int, float)
    radius:  radius of the disc (decimal degrees) (int, float)
    columns: [list] of column names to be returned in the result

    returns pandas.dataframe of entire catalog
    '''

    #vals = zip(
    #    df['catalog']
    #)
    vals = df['catalog'].values
    retdfs = []

    for catalog in vals:
        #try:
        df = pd.read_parquet(
            catalog, 
            engine='pyarrow',
            columns=columns
        )
        df["_DIST"]=util.gc_dist(
                df[c_md['ra_kw']], df[c_md['dec_kw']], ra, dec
        )
        df = df.loc[df['_DIST'] < radius]
        retdfs.append(df)
        del df
        #except:
        #    retdfs.append(pd.DataFrame({}, columns=columns))

    return pd.concat(retdfs)


def xmatch_from_daskdf(df, c1_md, c2_md, c1_cols, c2_cols, n_neighbors=1, dthresh=0.01):
    '''
    mapped function for calculating a cross_match between a partitioned dataframe 
     with columns [C1, C2, Order, Pix, ToCull1, ToCull2]
        C1 is the pathway to the catalog1.parquet file
        C2 is the pathway to the catalog2.parquet file
        Order is the healpix order
        Pix is the healpix pixel at the order for the calculation
        ToCull1 is a boolean which represents the original C1 file's healpix order > 
            than C2, thus the number of sources is potentiall 4 times greater than
            the number of sources in C2. We can optimize the comparison calculation
            by culling the sources from C1 that aren't in C2's order/pixel
        ToCull2 the same as ToCull1, but C2 order > C1
    '''

    vals = zip(
        df['C1'], 
        df['C2'],
        df['Order'],
        df['Pix'],
        df['ToCull1'],
        df['ToCull2']
    )

    #get the column names for the returning dataframe
    colnames = util.establish_pd_meta(c1_cols, c2_cols)
    retdfs = []

    #iterate over the partitioned dataframe
    #in theory, this should be just one dataframe
    # TODO: ensure that this just one entry in df, and remove the forloop
    for c1, c2, order, pix, tocull1, tocull2 in vals:

        # TODO: enforcemetadata=False
        # TODO: select columns in the pd.read_parquet(...) command
        # try/except is here because when enforcemetadata=True, it passes in 
        #  a test-dataframe that has values for filepaths as 'foo'/'bar'
        #  which breaks opening the pandas.read_parquet()
        #try:
        c1_df = pd.read_parquet(c1, engine='pyarrow')
        c2_df = pd.read_parquet(c2, engine='pyarrow')

        #if c1 and c2 columnames have the same column names
        # append a suffix _2 to the second catalog
        c2_md = util.cmd_rename_kws(c2_cols, c2_md)
        c2_df = util.frame_rename_cols(c2_df, cols=c2_cols)

        #get the center lon/lat of the healpix pixel
        (clon, clat) = hp.pix2ang(hp.order2nside(order), pix, nest=True, lonlat=True)

        #cull the catalog dataframes based on ToCull=True/False
        # and user defined columns
        # TODO: select columns in the pd.read_parquet(...) command
        c1_df = util.frame_cull(
            df=c1_df, df_md=c1_md,
            order=order, pix=pix,
            cols=c1_cols,
            tocull=tocull1
        )

        c2_df = util.frame_cull(
            df=c2_df, df_md=c2_md,
            order=order, pix=pix,
            cols=c2_cols,
            tocull=tocull2
        )
        
        #Sometimes the c1_df or c2_df contain zero sources 
        # after culling
        if len(c1_df) and len(c2_df):

            #calculate the xy gnomonic positions from the 
            # pixel's center for each dataframe
            xy1 = util.frame_gnomonic(c1_df, c1_md, clon, clat)
            xy2 = util.frame_gnomonic(c2_df, c2_md, clon, clat)

            #construct the KDTree from the comparative catalog: c2/xy2
            tree = KDTree(xy2, leaf_size=2)
            #find the indicies for the nearest neighbors 
            #this is the cross-match calculation
            dists, inds = tree.query(xy1, k=min([n_neighbors, len(xy2)]))

            #numpy indice magic for the joining of the two catalogs
            outIdx = np.arange(len(c1_df)*n_neighbors) # index of each row in the output table (0... number of output rows)
            leftIdx = outIdx // n_neighbors            # index of the corresponding row in the left table (0, 0, 0, 1, 1, 1, 2, 2, 2, ...)
            rightIdx = inds.ravel()                    # index of the corresponding row in the right table (22, 33, 44, 55, 66, ...)
            out = pd.concat(
                [
                    c1_df.iloc[leftIdx].reset_index(drop=True),   # select the rows of the left table
                    c2_df.iloc[rightIdx].reset_index(drop=True)   # select the rows of the right table
                ], axis=1)  # concat the two tables "horizontally" (i.e., join columns, not append rows)

            #save the order/pix/and distances for each nearest neighbor
            out['hips_k'] = order
            out['hips_pix'] = pix
            out["_DIST"] =util.gc_dist(
                out[c1_md['ra_kw']], out[c1_md['dec_kw']],
                out[c2_md['ra_kw']], out[c2_md['dec_kw']]
            )

            #cull the return dataframe based on the distance threshold
            out = out.loc[out['_DIST'] < dthresh]
            out = out[colnames]
            retdfs.append(out)
            #memory management
            del out, dists, inds, outIdx, leftIdx, rightIdx, xy1, xy2

        else:
            retdfs.append(pd.DataFrame({}, columns=colnames))
        del c1_df, c2_df

        #except:
        #    retdfs.append(pd.DataFrame({}, columns=colnames))

    #if the npartitions are > 1 it will concatenate calculated catalogcrossmatches into a single dataframe to return
    return pd.concat(retdfs)


if __name__ == '__main__':
    import time
    s = time.time()
    #client = Client(n_workers=12, threads_per_worker=1)
    print('runnin')
    parts = ['http://cdn.gea.esac.esa.int/Gaia/gdr3/gaia_source/GaiaSource_000000-003111.csv.gz']
    k = 10
    cache_dir = os.path.join(os.getcwd(), 'cache', 'gaia_exA')
    fmt = 'csv.gz'
    ra_kw = 'ra'
    dec_kw = 'dec'
    skiprows = np.arange(0,1000)
    _gather_statistics_hpix_hist(parts=parts, k=k, cache_dir=cache_dir, fmt=fmt, ra_kw=ra_kw, dec_kw=dec_kw, skiprows=skiprows)
    #parts, k, cache_dir, fmt, ra_kw, dec_kw, skiprows=None
    e = time.time()
    print('Elapsed time = {}'.format(e-s))