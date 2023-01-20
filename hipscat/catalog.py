import os
import sys
import glob
import json

import numpy as np
import healpy as hp
from dask.distributed import Client, progress
from matplotlib import pyplot as plt 
import dask.dataframe as dd
import pandas as pd

from . import util
from . import dask_utils as du
from . import partitioner as pt

'''

user experience:

    from hipscat import catalog as cat
    gaia = cat.Catalog('gaia')

    -> checks in source location for the 'hierarchical formatted data'

    -> if it isn't located in source notify user that the catalog must be formatted

    -> local formatting gaia.hips_import(dir='/path/to/catalog/', fmt='csv.gz') [1]
        this outputs to local directory: output/ (crib mario's code for gaia 1st 10 files)


'''
class Catalog():

    def __init__(self, catname='gaia', source='local', location='/epyc/projects3/sam_hipscat/'):

        self.catname = catname
        self.source = source
        self.hips_metadata = None
        self.partitioner = None
        self.output_dir = None
        self.result = None
        self.location = location
        self.lsddf = None

        if self.source == 'local':
            self.output_dir = os.path.join(self.location, 'output', self.catname)
            if not os.path.exists(self.output_dir):
                print('No local hierarchal catalog exists, run catalog.hips_import(file_source=\'/path/to/file_or_files\', fmt=\'csv.gz\')')
            else:
                metadata_file = os.path.join(self.output_dir, f'{self.catname}_meta.json')

                if os.path.exists(metadata_file):
                    print(f'Located Partitioned Catalog: {metadata_file}')
                    with open(metadata_file) as f:
                        self.hips_metadata = json.load(f)
                else:
                    print('Catalog not fully imported. Re-run catalog.hips_import()')

        elif self.source == 's3':
            sys.exit('Not Implemented ERROR')
        else:
            sys.exit('Not Implemented ERROR')


    def __repr__(self):
        return f"Catalog({self.catname})"


    def __str__(self):
        return f"Catalog: {self.catname}"


    def load(self, columns=None):
        #dirty way to load as a dask dataframe
        assert self.hips_metadata is not None, 'Catalog has not been partitioned!'
        hipscat_dir = os.path.join(self.output_dir, 'Norder*', 'Npix*', 'catalog.parquet')

        self.lsddf = dd.read_parquet(
            hipscat_dir,
            calculate_divisions=True,
            columns=columns
        )
        return self.lsddf


    def hips_import(self, file_source='/data2/epyc/data/gaia_edr3_csv/', 
        fmt='csv.gz', debug=False, verbose=True, limit=None, threshold=1_000_000, 
        ra_kw='ra', dec_kw='dec', id_kw='source_id', client=None):

        '''
            ingests a list of source catalog files and partitions them out based 
            on hierarchical partitioning of size on healpix map

            supports http list of files, s3 bucket files, or local files
            formats currently supported: csv.gz, csv, fits, parquet
        '''

        if 'http' in file_source:
            urls = util.get_csv_urls(url=file_source, fmt=fmt)

        elif 's3' in file_source:
            sys.exit('Not Implemented ERROR')

        else: #assume local?
            if os.path.exists(file_source):
                fs_clean = file_source
                if fs_clean[-1] != '/':
                    fs_clean += '/'
                urls = glob.glob('{}*{}'.format(fs_clean, fmt))
            else:
                sys.exit('Local files not found at source {}'.format(file_source))

        if limit:
            urls = urls[0:limit]

        if verbose:
            print(f'Attempting to format files: {len(urls)}')

        if len(urls):
            self.partitioner = pt.Partitioner(catname=self.catname, fmt=fmt, urls=urls, id_kw=id_kw,
                        order_k=10, verbose=verbose, debug=debug, ra_kw=ra_kw, dec_kw=dec_kw, 
                        location=self.location)

            if debug:
                self.partitioner.gather_statistics()
                self.partitioner.compute_partitioning_map(max_counts_per_partition=threshold)
                self.partitioner.write_structure_metadata()
            else:
                self.partitioner.run(client=client, threshold=threshold)

        else:
            print('No files Found!')


    def cone_search(self, ra, dec, radius, columns=None):
        '''
        Perform a cone search on HiPSCat

        Parameters:
         ra:     float, Right Ascension
         dec:    float, Declination
         radius: float, Radius from center that circumvents, must be in degrees
        '''
        assert self.hips_metadata is not None, f'{self} hipscat metadata not found. {self}.hips_import() needs to be (re-) ran'
        assert isinstance(ra, (int, float))
        assert isinstance(dec, (int, float))
        assert isinstance(radius, (int, float))
        
        #establish metadata for the returning dask.dataframe
        # user can select columns
        # returns a distance column for proximity metric

        ddf = self.load(columns=columns)
        meta = ddf._meta
        meta['_DIST'] = []
        
        #utilize the healpy library to find the pixels that exist
        # within the cone. If user-specified radius is too small
        # we iterate over increasing orders, until we find
        # pixels to query at the order.
        highest_order, pixels_to_query = util.find_pixels_in_disc_at_efficient_order(ra, dec, radius)
        
        #query our hips metadata for partitioned pixel catalogs that 
        # lie within the cone
        cone_search_map = []
        for p in pixels_to_query:
            mapped_dict = util.map_pixel_at_order(int(p), highest_order, self.hips_metadata['hips'])
            for mo in mapped_dict:
                mapped_pixels = mapped_dict[mo]
                for mp in mapped_pixels:
                    cat_path = os.path.join(self.output_dir, f'Norder{mo}', f'Npix{mp}', 'catalog.parquet')
                    cone_search_map.append(cat_path)

        #only need a catalog once, remove duplicates
        # and then establish the create the dask.dataframe
        # that will perform the map_partitions of the cone_search
        # the only parameter we need in this dataframe is the catalog pathways
        cone_search_map = list(set(cone_search_map))
        cone_search_map_dict = {
            'catalog':cone_search_map
        }

        nparts = len(cone_search_map_dict[list(cone_search_map_dict.keys())[0]])
        if nparts > 0:
            cone_search_df = dd.from_pandas(
                pd.DataFrame(
                    cone_search_map_dict, 
                    columns=list(cone_search_map_dict.keys())
                ).reset_index(drop=True), 
                npartitions=nparts
            )

            #calculate the sources with the dask partitinoed ufunc
            result = cone_search_df.map_partitions(
                du.cone_search_from_daskdf,
                ra=ra, dec=dec, radius=radius, 
                c_md=self.hips_metadata, columns=columns,
                meta=meta
            )   
            return result
        else:
            #No sources in the catalog within the disc
            return dd.from_pandas(meta, npartitions=1)


    def cross_match(self, othercat=None, c1_cols={}, c2_cols={}, n_neighbors=1, dthresh=0.01, debug=False):
        '''
            Parameters:
                othercat- other hipscat catalog

                user-defined columns to return for dataframe
                c1_cols- dictionary of {column_name : dtype}
                c2_cols- dictionary of {column_name : dtype}
                    dtypes -> f8 - float, i9 - int, etc
                n_neighbors - number of nearest neighbors to find for each souce in catalog1
                dthresh- distance threshold for nearest neighbors (decimal degrees)
        '''

        assert othercat is not None, 'Must specify another catalog to crossmatch with.'
        assert isinstance(othercat, Catalog), 'The other catalog must be an instance of hipscat.Catalog.'

        assert self.hips_metadata is not None, f'{self} hipscat metadata not found. {self}.hips_import() needs to be (re-) ran'
        assert othercat.hips_metadata is not None, f'{othercat} hipscat metadata not found. {othercat}.hips_import() needs to be (re-) ran'

        #Gather the metadata from the already partitioned catalogs
        cat1_md = self.hips_metadata
        cat2_md = othercat.hips_metadata
        
        #use the metadata to calculate the hipscat1 x hipscat2 map
        # this function finds the appropriate catalog parquet files to execute the crossmatches
        # returns a [
        #   [path/to/hipscat1/catalog.parquet, path/to/hipscat2/catalog.parquet]
        # ]
        hc_xmatch_map = util.map_catalog_hips(cat1_md['hips'], self.output_dir,
                cat2_md['hips'], othercat.output_dir)


        if debug:
            hc_xmatch_map = hc_xmatch_map[:5]
            print('DEBUGGING ONLY TAKING 5')
        
        #This instantiates the dask.dataframe from the hc
        #  just a table with columns = [catalog1_file_path, catalog2_file_path, other_xm_metadata...] 
        matchcats_dict =util.xmatchmap_dict(hc_xmatch_map)
        # ensure the number of partitions are the number cross-match operations so that memory is managed
        nparts = len(matchcats_dict[list(matchcats_dict.keys())[0]])

        matchcats_df = dd.from_pandas(
            pd.DataFrame(
                matchcats_dict, 
                columns = list(matchcats_dict.keys())
            ).reset_index(drop=True), 
            npartitions=nparts
        )

        #establish the return columns for the returned dataframe's metadata
        # dask.dataframe.map_partitions() requires the metadata of the resulting 
        # dataframe to be defined prior to execution. The column names and datatypes
        # are defined here and passed in the 'meta' variable
        c1_cols = util.catalog_columns_selector_withdtype(cat1_md, c1_cols)
        c2_cols = util.catalog_columns_selector_withdtype(cat2_md, c2_cols)
        c2_cols = util.rename_meta_dict(c1_cols, c2_cols)

        #populate metadata with column information 
        # plus variables from the cross_match calculation
        meta = {}
        meta.update(c1_cols)
        meta.update(c2_cols)
        meta.update({
            'hips_k':'i8', 
            'hips_pix':'i8',
            '_DIST':'f8'
        })

        #call the xmatch_from_daskdf function.
        self.result = matchcats_df.map_partitions(
            du.xmatch_from_daskdf,
            cat1_md, cat2_md, 
            list(c1_cols.keys()), list(c2_cols.keys()),
            n_neighbors=n_neighbors,
            dthresh=dthresh,
            meta = meta
        )
        return self.result
    

    def visualize_sources(self, figsize=(5,5)):

        if self.output_dir is not None:
            outputdir_files = os.listdir(self.output_dir)
            maps = [x for x in outputdir_files if f'{self.catname}_order' in x and 'hpmap.fits' in x]
            if len(maps):
                mapFn = os.path.join(self.output_dir, maps[0])
                img = hp.read_map(mapFn)
                fig = plt.figure(figsize=figsize)
                return hp.mollview(np.log10(img+1), fig=fig, title=f'{self.catname}: {img.sum():,.0f} sources', nest=True)
            else:
                assert False, 'map not found. hips_import() needs to be (re-) ran'
        else:
            assert False, 'hipscat output_dir not found. hips_import() needs to be (re-) ran'


    def visualize_partitions(self, figsize=(5,5)):

        if self.hips_metadata is not None:
            catalog_hips = self.hips_metadata["hips"]
            k = max([int(x) for x in catalog_hips.keys()])
            npix = hp.order2npix(k)
            orders = np.full(npix, -1)
            idx = np.arange(npix)
            c_orders = [int(x) for x in catalog_hips.keys()]
            c_orders.sort()
            
            for o in c_orders:
                k2o = 4**(k-o)
                pixs = catalog_hips[str(o)]
                pixk = idx.reshape(-1, k2o)[pixs].flatten()
                orders[pixk] = o
            
            fig = plt.figure(figsize=figsize)
            return hp.mollview(orders, fig=fig, max=k, title=f'{self.catname} partitions', nest=True)
        else:
            assert False, 'hipscat metadata not found. hips_import() needs to be (re-) ran'
            
        
    def visualize_cone_search(self, ra, dec, radius, figsize=(5,5)):
        assert self.hips_metadata is not None, f'{self} hipscat metadata not found. {self}.hips_import() needs to be (re-) ran'
        assert isinstance(ra, (int, float))
        assert isinstance(dec, (int, float))
        assert isinstance(radius, (int, float))

        highest_order = 10
        nside = hp.order2nside(highest_order)
        vec = hp.ang2vec(ra, dec, lonlat=True)
        rad = np.radians(radius)
        pixels_to_query = hp.query_disc(nside, vec, rad, nest=True)

        if self.output_dir is not None:
            outputdir_files = os.listdir(self.output_dir)
            maps = [x for x in outputdir_files if f'{self.catname}_order' in x and 'hpmap.fits' in x]
            if len(maps):
                map0 = maps[0]
                mapFn = os.path.join(self.output_dir, map0)
                highest_order = int(mapFn.split('order')[1].split('_')[0])
                nside = hp.order2nside(highest_order)
                vec = hp.ang2vec(ra, dec, lonlat=True)
                rad = np.radians(radius)
                pixels_to_query = hp.query_disc(nside, vec, rad, nest=True)

                img = hp.read_map(mapFn)
                maxN = max(np.log10(img+1))+1
                img[pixels_to_query] = maxN
                fig = plt.figure(figsize=figsize)
                return hp.mollview(np.log10(img+1), fig=fig, title=f'Cone search of {self.catname}', nest=True)
            else:
                assert False, 'map not found. hips_import() needs to be (re-) ran'
        else:
            assert False, 'hipscat output_dir not found. hips_import() needs to be (re-) ran'
        

    def visualize_cross_match(self, othercat, figsize=(5,5)):
        '''
            visualize the overlap when crossmatching two catalogs
        '''
        sys.exit('Not implemented error')


if __name__ == '__main__':
    import time
    s = time.time()
    ###
    #cat = Catalog()
    ###
    e = time.time()
    print("Elapsed time: {}".format(e-s))
