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
    from . import lsd2_io
except ImportError:
    import util
    import lsd2_io

from . import margin_utils

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


def _write_partition_structure(url, cache_dir, output_dir, orders, opix, ra_kw, dec_kw, id_kw, neighbor_pix, highest_k, margin_threshold, calculate_neighbors, verbose=False):

    #order_kw = 'hips_k'
    #pix_kw = 'hips_pix'
    #HACK
    order_kw = 'Norder'
    pix_kw = 'Npix'
    dir_kw = 'Dir'

    base_filename = os.path.basename(url).split('.')[0]
    parqFn = os.path.join(cache_dir, base_filename + '.parquet')
    df = pd.read_parquet(parqFn, engine='pyarrow')
    df['tmp_uniq'] = np.arange(len(df))

    #hack if there isn't a header
    if all([isinstance(x, int) for x in [ra_kw, dec_kw, id_kw]]):
        ra_kw = df.keys()[ra_kw]
        dec_kw = df.keys()[dec_kw]
        id_kw = df.keys()[id_kw]

    if calculate_neighbors:
        # write the margin data
        df['margin_pix'] = hp.ang2pix(2**highest_k, df[ra_kw].values, df[dec_kw].values, lonlat=True, nest=True)
        
        neighbor_cache_df = df.merge(neighbor_pix, on='margin_pix')
        neighbor_cache_df[dir_kw] = (neighbor_cache_df['part_pix'] / 10_000) * 10_000

        convert_dict = {
            'part_pix': np.int32,
            'part_order': np.int32,
            dir_kw : np.int32
        }
        neighbor_cache_df = neighbor_cache_df.astype(convert_dict)

        res = neighbor_cache_df.groupby(['part_pix', 'part_order'], group_keys=False).apply(
            _to_neighbor_cache, 
            hipsPath=output_dir, 
            base_filename=base_filename, 
            ra_kw=ra_kw, 
            dec_kw=dec_kw,
            highest_k=highest_k,
            margin_threshold=margin_threshold
        )

        del neighbor_cache_df

    for k in orders:
        df[order_kw] = k
        df[pix_kw] = hp.ang2pix(2**k, df[ra_kw].values, df[dec_kw].values, lonlat=True, nest=True)
        df[dir_kw] = (df[pix_kw] / 10_000) * 10_000

        convert_dict = {
            order_kw: np.int32,
            pix_kw: np.int32,
            dir_kw: np.int32
        }
        df = df.astype(convert_dict)
        order_df = df.loc[df[pix_kw].isin(opix[k])]

        #audit_counts[k].append(len(order_df))

        #reset the df so that it doesn't include the already partitioned sources
        # ~df['column_name'].isin(list) -> sources not in order_df sources
        df = df.loc[~df['tmp_uniq'].isin(order_df['tmp_uniq'])]
        
        #groups the sources in order_k pixels, then outputs them to the base_filename sources
        ret = order_df.groupby([order_kw, pix_kw], group_keys=False).apply(_to_hips, hipsPath=output_dir, base_filename=base_filename)

        del order_df
        if len(df) == 0:
            break

    return 0


def _map_reduce(tmp_pixel_dir, filename, ra_kw, dec_kw, dtypes, indexing_order=14):
    #print(output_dirs)
    #for output_dir in output_dirs:
    '''
        pixel_dir -> /path/to/catalog/Norder=N/Npix=P/[parquet_files]
    '''

    catalog_dir = tmp_pixel_dir.split('Norder=')[0]
    parq_files = os.listdir(os.path.join(tmp_pixel_dir))

    #so it doesn't re-concatenate the original catalog/neighbor.parquet, if partition isn't re-ran
    #parq_files = list(filter(lambda f: len(f) > len(filename) and f[-len(filename):] == filename, files))

    if len(parq_files) == 1:
        fn = os.path.join(tmp_pixel_dir, parq_files[0])
        norder, npix = lsd2_io.get_norder_npix_from_tmpdir(fn)

        df = pd.read_parquet(fn, engine='pyarrow')
        df["_ID"] = util.compute_index(df[ra_kw].values, df[dec_kw].values, order=indexing_order)
        df.set_index("_ID", inplace=True)
        df.sort_index(inplace=True)

        if dtypes is not None:
            df = df.astype(dtypes)

        pixel_dir = lsd2_io.get_hipscat_pixel_dir(norder, npix)
        os.makedirs(os.path.join(catalog_dir, pixel_dir), exist_ok=True)

        pixel_file = lsd2_io.get_hipscat_pixel_file(norder, npix)
        df.to_parquet(os.path.join(catalog_dir, pixel_file))
        del df

    if len(parq_files) > 1:
        df_files = [os.path.join(tmp_pixel_dir, f) for f in parq_files]
        norder, npix = lsd2_io.get_norder_npix_from_tmpdir(df_files[0])

        df = pd.concat(
            [pd.read_parquet(parq_file, engine='pyarrow')
            for parq_file in df_files], sort=False
        )

        df["_ID"] = util.compute_index(df[ra_kw].values, df[dec_kw].values, order=indexing_order)
        df.set_index("_ID", inplace=True)
        df.sort_index(inplace=True)

        if dtypes is not None:
            df = df.astype(dtypes)

        pixel_dir = lsd2_io.get_hipscat_pixel_dir(norder, npix)
        os.makedirs(os.path.join(catalog_dir, pixel_dir), exist_ok=True)

        pixel_file = lsd2_io.get_hipscat_pixel_file(norder, npix)
        df.to_parquet(os.path.join(catalog_dir, pixel_file))
        del df

    os.system(f"rm -rf {tmp_pixel_dir}")
    return 0


def _to_hips(df, hipsPath, base_filename):

    #order_kw = 'hips_k'
    #pix_kw = 'hips_pix'
    #HACK
    order_kw = 'Norder'
    pix_kw = 'Npix'
    # WARNING: only to be used from df2hips(); it's defined out here just for debugging
    # convenience.

    # grab the order and pix number for this dataframe. Since this function
    # is intented to be called with apply() after running groupby on (k, pix), these must
    # be the same throughout the entire dataframe
    k = df[order_kw].iloc[0]
    pix = df[pix_kw].iloc[0]
    
    assert (df[order_kw]   ==   k).all()
    assert (df[pix_kw] == pix).all()

    # construct the output directory and filename
    #HACK
    #dir = os.path.join(hipsPath, f'Norder{k}/Npix{pix}')
    dir = os.path.join(hipsPath, f'catalog/Norder={k}/Npix={pix}')
    fn  = os.path.join(dir, f'{base_filename}_catalog.parquet')

    # create dir if it doesn't exist
    os.makedirs(dir, exist_ok=True)

    # write to the file (append if it already exists)
    # also, write the header only if the file doesn't already exist
    drop_columns=['tmp_uniq','margin_pix'] #, 'hips_k', 'hips_pix']
    df = df.drop(columns=drop_columns, axis=1)
    df.to_parquet(fn)

    # return the number of records written
    del df
    return 0

def _to_neighbor_cache(df, hipsPath, base_filename, ra_kw, dec_kw, highest_k, margin_threshold):
    # WARNING: only to be used from df2hips(); it's defined out here just for debugging
    # convenience.

    # grab the order and pix number for this dataframe. Since this function
    # is intented to be called with apply() after running groupby on (k, pix), these must
    # be the same throughout the entire dataframe
    k, pix = df['part_order'].iloc[0],  df['part_pix'].iloc[0]
    assert (df['part_pix'] == pix).all()
    assert (df['part_order']   ==   k).all()

    scale = margin_utils.get_margin_scale(k, margin_threshold)

    # create the rough boundaries of the threshold bounding region.
    bounding_polygons = margin_utils.get_margin_bounds_and_wcs(k, pix, scale)

    is_polar, pole = margin_utils.is_polar(k, pix)

    if is_polar:
        trunc_pix = margin_utils.get_truncated_pixels(k, pix, highest_k, pole)
        df['is_trunc'] = np.isin(df['margin_pix'], trunc_pix)

        trunc_data = df.loc[df['is_trunc'] == True]
        other_data = df.loc[df['is_trunc'] == False]

        trunc_data['margin_check'] = margin_utils.check_polar_margin_bounds(
            trunc_data[ra_kw].values,
            trunc_data[dec_kw].values,
            k,
            pix,
            highest_k,
            pole,
            margin_threshold
        )
        other_data['margin_check'] = margin_utils.check_margin_bounds(
            other_data[ra_kw].values, 
            other_data[dec_kw].values, 
            bounding_polygons
        )

        df = pd.concat([trunc_data, other_data])
    else:
        df['margin_check'] = margin_utils.check_margin_bounds(
            df[ra_kw].values, 
            df[dec_kw].values, 
            bounding_polygons
        )

    margin_df = df.loc[df['margin_check'] == True]

    if len(margin_df):
        # construct the output directory and filename
        dir = os.path.join(hipsPath, f'neighbor/Norder={k}/Npix={pix}')
        fn  = os.path.join(dir, f'{base_filename}_neighbor.parquet')
        # create dir if it doesn't exist
        os.makedirs(dir, exist_ok=True)

        # write to the file (append if it already exists)
        # also, write the header only if the file doesn't already exist
        rename_dict = {
            'part_pix' : 'Npix',
            'part_order' : 'Norder'
        }
        drop_columns=['tmp_uniq','margin_pix', 'margin_check']
        margin_df = margin_df.drop(columns=drop_columns, axis=1).rename(rename_dict, axis=1)
        margin_df.to_parquet(fn)

        # return the number of records written
        del df, margin_df
        return 0
    else:
        del df, margin_df
        return 0


def _check_margin_bounds(ra, dec, pixel_region):
    res = []
    for i in range(len(ra)):
        sc = PixCoord(x=ra[i], y=dec[i])
        in_bounds = pixel_region.contains(sc)
        res.append(in_bounds)
    return res


def cone_search_from_daskdf(df, c_md, ra, dec, radius, columns=None, storage_options=None):
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
        df = lsd2_io.read_parquet(catalog, 'pandas', columns=columns, storage_options=storage_options)
        df["_DIST"]=util.gc_dist(
                df[c_md['ra_kw']], df[c_md['dec_kw']], ra, dec
        )
        df = df.loc[df['_DIST'] < radius]
        retdfs.append(df)
        del df
        #except:
        #    retdfs.append(pd.DataFrame({}, columns=columns))

    return pd.concat(retdfs)


def xmatch_from_daskdf(df, all_column_dict, n_neighbors=1, dthresh=0.01, evaluate_margins=True, storage_options=None):
    '''
    mapped function for calculating a cross_match between a partitioned dataframe
     with columns [C1, C2, Order, Pix]
        C1 is the pathway to the catalog1.parquet file
        C2 is the pathway to the catalog2.parquet file
        Order is the healpix order
        Pix is the healpix pixel at the order for the calculation
    '''

    vals = zip(
        df['C1'],
        df['C2'],
        df['Order'],
        df['Pix']
    )

    #get the column names for the returning dataframe
    colnames = []
    colnames.extend(all_column_dict['c1_cols_prefixed'])
    colnames.extend(all_column_dict['c2_cols_prefixed'])
    colnames.extend([
        'hips_k',
        'hips_pix',
        '_DIST'
    ])
    retdfs = []

    #get the prefixed kw metadata for each catalog
    c1_md = all_column_dict['c1_kws_prefixed']
    c2_md = all_column_dict['c2_kws_prefixed']

    #iterate over the partitioned dataframe
    #in theory, this should be just one dataframe
    # TODO: ensure that this just one entry in df, and remove the forloop
    for c1, c2, order, pix in vals:
        
        #implement neighbors into xmatch routine
        # exists at the same path as c1/c2, 
        # just has neighbor instead of catalog in the filename
        # We should only consider neighbors margins from cat2 - SW 02/16/2023
        
        #c2 = /some/path/hipscat/catalog/Norder=N/Dir=D/Npix=N.parquet
        #c2_split = ['/some/path/hipscat', 'Norder=N/Dir=D/Npix=N.parquet']
        #n2 = '/some/path/hipscat/neighbor/Norder=N/Dir=D/Npix=N.parquet' :)
        c2_split = c2.split('/catalog/')
        n2 = os.path.join(c2_split[0], 'neighbor', c2_split[1])
        
        # read the cat1_df while culling appropriate columns
        c1_df = lsd2_io.read_parquet(c1, 'pandas', columns=all_column_dict['c1_cols_original'], storage_options=storage_options)
        c1_df.columns = all_column_dict['c1_cols_prefixed']

        #cat2_df = pd.read_parquet(c2, columns=all_column_dict['c2_cols_original'], engine='pyarrow')
        cat2_df = lsd2_io.read_parquet(c2, 'pandas', columns=all_column_dict['c2_cols_original'], storage_options=storage_options)
        # if a neighbor file exists
        if (lsd2_io.file_exists(n2) and evaluate_margins):
            # read
            n2_df = lsd2_io.read_parquet(n2, 'pandas', hipsdir='neighbor', columns=all_column_dict['c2_cols_original'], storage_options=storage_options)
            # and concatenate
            c2_df = pd.concat([cat2_df, n2_df])
        else:
            c2_df = cat2_df
        # rename the columns with appropriate prefixes
        c2_df.columns = all_column_dict['c2_cols_prefixed']

        #get the center lon/lat of the healpix pixel
        (clon, clat) = hp.pix2ang(hp.order2nside(order), pix, nest=True, lonlat=True)
        
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
            out['hips_k']   = order
            out['hips_pix'] = pix
            out["_DIST"]    = util.gc_dist(
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
    #tests
    e = time.time()
    print('Elapsed time = {}'.format(e-s))
