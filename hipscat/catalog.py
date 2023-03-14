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
import warnings

from . import util
from . import dask_utils as du
from . import partitioner as pt
from . import lsd2_io


class Catalog():

    def __init__(self, catname='gaia', location='/epyc/projects3/sam_hipscat/', storage_options=None):

        self.catname = catname
        self.location = location
        self.storage_options = storage_options
        self.hips_metadata = None
        self.partitioner = None
        self.output_dir = None
        self.result = None
        self.lsddf = None

        self.output_dir = os.path.join(self.location, self.catname)
        if not lsd2_io.file_exists(self.output_dir, self.storage_options):
            raise FileNotFoundError(f"Location to HiPSCatalog does not exist")
        
        self.hips_metadata = lsd2_io.read_hipsmeta_file(self.output_dir, self.storage_options)


    def __repr__(self):
        return f"Catalog({self.catname})"


    def __str__(self):
        return f"Catalog: {self.catname}"


    def load(self, columns=None):
        #dirty way to load as a dask dataframe
        assert self.hips_metadata is not None, 'Catalog has not been partitioned!'

        #ensure user doesn't pass in empty list
        columns = columns if (isinstance(columns, list) and len(columns) > 0) else None

        self.lsddf = lsd2_io.read_parquet(
            pathway=self.output_dir,
            library='dask',
            engine='pyarrow',
            columns=columns,
            storage_options=self.storage_options
        )
        return self.lsddf


    def cone_search(self, ra, dec, radius, columns=None):
        '''
        Perform a cone search on HiPSCat

        Parameters:
         ra:     float, Right Ascension
         dec:    float, Declination
         radius: float, Radius from center that circumvents, must be in degrees
        '''
        assert self.hips_metadata is not None, f'{self} hipscat metadata not found. {self}.hips_import() needs to be (re-) ran'
        assert isinstance(ra, (int, float)), f'ra must be a number'
        assert isinstance(dec, (int, float)), f'dec must be a number'
        assert isinstance(radius, (int, float)), f'radius must be a number'
        
        #establish metadata for the returning dask.dataframe
        # user can select columns
        # returns a distance column for proximity metric

        ddf = self.load(columns=util.validate_user_input_cols(columns, self.hips_metadata))
        meta = ddf._meta

        if columns is None and all(x in meta.columns for x in ['dir0', 'dir1']):
            meta = meta.drop(columns=['dir0', 'dir1'], axis=1)

        meta['_DIST'] = []
        
        #utilize the healpy library to find the pixels that exist
        # within the cone. 
        vec = hp.ang2vec(ra, dec, lonlat=True)
        rad = np.radians(radius)
        highest_order = int(max(self.hips_metadata['hips'].keys()))
        nside = hp.order2nside(highest_order)
        pixels_to_query = hp.query_disc(nside, vec, rad, nest=True, inclusive=True)
        
        #query our hips metadata for partitioned pixel catalogs that 
        # lie within the cone
        cone_search_map = []
        for p in pixels_to_query:
            mapped_dict = util.map_pixel_at_order(int(p), highest_order, self.hips_metadata['hips'])
            for mo in mapped_dict:
                mapped_pixels = mapped_dict[mo]
                for mp in mapped_pixels:
                    cat_path = os.path.join(self.output_dir, 'catalog', lsd2_io.HIPSCAT_DIR_STRUCTURE.format(mo, mp, 'catalog'))
                    cone_search_map.append(cat_path)

        #only need a catalog once, remove duplicates
        # and then create the dask.dataframe that will perform 
        # the map_partitions of the cone_search the only parameter 
        # we need in this dataframe is the catalog pathways
        cone_search_map = list(set(cone_search_map))
        cone_search_map_dict = {
            'catalog':cone_search_map
        }

        #nparts = len(cone_search_map_dict[list(cone_search_map_dict.keys())[0]])
        nparts = len(cone_search_map)
        if nparts == 0:
            #No sources in the catalog within the disc
            return dd.from_pandas(meta, npartitions=1)

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
            storage_options=self.storage_options,
            meta=meta
        )   
        return result


    def cross_match(self, othercat=None, c1_cols=[], c2_cols=[], n_neighbors=1, dthresh=0.01, evaluate_margins=True, debug=False):
        '''
            Parameters:
                othercat- other hipscat catalog

                user-defined columns to return for dataframe
                c1_cols- list of [column_name]
                c2_cols- list of [column_name]
                n_neighbors - number of nearest neighbors to find for each souce in catalog1
                dthresh- distance threshold for nearest neighbors (decimal degrees)
        '''

        assert othercat is not None, 'Must specify another catalog to crossmatch with.'
        assert isinstance(othercat, Catalog), 'The other catalog must be an instance of hipscat.Catalog.'
        assert self.catname != othercat.catname, "Cannot cross_match catalog with self"
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

        #establish the return columns for the returned xmatch'd dataframe's metadata
        # dask.dataframe.map_partitions() requires the metadata of the resulting 
        # dataframe to be defined prior to execution. The column names are defined here 
        # renamed based on catalog_name, and passed into the 'meta' variable

        # lazy load the dataframes to 'sniff' the column names. Doesn't load any data
        # The util.validate_user_input_cols function just ensures the ra, dec, and id columns aren't forgotten
        # if the user doesn't specify columns, it will grab the whole dataframe
        c1_ddf = self.load(columns=util.validate_user_input_cols(c1_cols, cat1_md))
        c2_ddf = othercat.load(columns=util.validate_user_input_cols(c2_cols, cat2_md))

        # grab the meta data for each catalogs
        c1_meta = c1_ddf._meta
        c2_meta = c2_ddf._meta

        #rename the meta-data with the catalog name prefixes
        c1_meta_prefixed = util.frame_prefix_all_cols(c1_meta, self.catname)
        c2_meta_prefixed = util.frame_prefix_all_cols(c2_meta, othercat.catname)

        #construct the dictionary that the cross_match routine utilizes to 
        # - read the specified columns from the parquet file 'c1/2_cols_original'
        # - rename the columns with the prefix names for the return dataframe 'c1/2_cols_prefixed'
        # - maintain the prefixed kws for ra/dec/id so that the cross_match routine can 
        #       appropriately read them from the dataframes when actually cross_matching
        all_column_dict = {
            'c1_cols_original' : list(c1_meta.columns),
            'c1_cols_prefixed' : list(c1_meta_prefixed.columns),
            'c2_cols_original' : list(c2_meta.columns),
            'c2_cols_prefixed' : list(c2_meta_prefixed.columns),
            'c1_kws_prefixed'  : util.catalog_prefix_kws(cat1_md, self.catname),    #rename the ra,dec,id kws with 
            'c2_kws_prefixed'  : util.catalog_prefix_kws(cat2_md, othercat.catname) # the appropriate prefixes
        }
        del c1_ddf, c2_ddf

        # TODO: Do we want to allow same catalog crossmatching, requires _2 suffix?
        # c2_cols = util.rename_meta_cols(c1_cols, c2_cols)

        #create the empty metadata dataframe along with our xmatch columns
        meta = pd.concat([c1_meta_prefixed, c2_meta_prefixed])
        meta['hips_k'] = []
        meta['hips_pix'] = []
        meta['_DIST'] = []

        #let the user know they are about to get huge dataframes
        if len(meta.columns) > 50:
            warnings.warn('The number of columns in the returned dataframe is greater than 50. \n \
                This could potentially excede the expected computation time. \n \
                It is highly suggested to limit the return columns for the cross_match with the c1_cols=[...], and c2_cols=[...] parameters'
            )

        #call the xmatch_from_daskdf ufunction.
        self.result = matchcats_df.map_partitions(
            du.xmatch_from_daskdf,
            all_column_dict=all_column_dict,
            n_neighbors=n_neighbors,
            dthresh=dthresh,
            evaluate_margins=evaluate_margins,
            storage_options=self.storage_options,
            meta=meta
        )
        return self.result


    def visualize_sources(self, figsize=(5,5)):
        '''
        Returns hp.mollview() of the high order pixel map that is 
        calculated during the partitioning routine. 
        
        inputs:
            figsize=Tuple(x,y) for the figure size

        Visualize from notebook    
        '''

        img = lsd2_io.read_fits_file(self.output_dir, self.storage_options)
        fig = plt.figure(figsize=figsize)
        return hp.mollview(np.log10(img+1), fig=fig, title=f'{self.catname}: {img.sum():,.0f} sources', nest=True)


    def visualize_partitions(self, figsize=(5,5)):
        '''
        Returns hp.mollview() of the partitioning structure that is 
        calculated during the partitioning routine. 
        
        inputs: 
            figsize=Tuple(x,y) for the figure size

        Visualize from notebook
        '''
        assert self.hips_metadata is not None, f'{self} hipscat metadata not found. {self}.hips_import() needs to be (re-) ran'

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
            
        
    def visualize_cone_search(self, ra, dec, radius, figsize=(5,5)):
        '''
        Returns hp.mollview() of a cone-search 
        
        inputs: 
            ra=float, Right Ascension in decimal degrees
            dec=float, Declination in decimal degrees
            radius=float, Radius of cone in degrees
            figsize=Tuple(x,y), for the figure size

        Visualize from notebook
        '''
        assert self.hips_metadata is not None, f'{self} hipscat metadata not found. {self}.hips_import() needs to be (re-) ran'
        assert isinstance(ra, (int, float)), f'ra must be a number'
        assert isinstance(dec, (int, float)), f'dec must be a number'
        assert isinstance(radius, (int, float)), f'radius must be a number'

        #grab the required parameters for hp.query(disc)
        highest_order = 10
        nside = hp.order2nside(highest_order)
        vec = hp.ang2vec(ra, dec, lonlat=True)
        rad = np.radians(radius)
        pixels_to_query = hp.query_disc(nside, vec, rad, nest=True)

        img = lsd2_io.read_fits_file(self.output_dir, self.storage_options)
        maxN = max(np.log10(img+1))+1

        #plot pencil beam. Maybe change the maxN to a value that will 
        #   never occur in np.log10(img+1).. maybe a negative number
        img[pixels_to_query] = maxN
        fig = plt.figure(figsize=figsize)
        return hp.mollview(np.log10(img+1), fig=fig, title=f'Cone search of {self.catname}', nest=True)


    def visualize_cross_match(self, othercat, figsize=(5,5)):
        '''
            Returns hp.mollview of the overlap when crossmatching two catalogs

            inputs:
                othercat=hipscat.Catalog()
                figsuze=Tuple(x,y)

        Visualize from notebook
        '''
        raise NotImplementedError('catalog.visualize_cross_match() not implemented yet')


if __name__ == '__main__':
    import time
    s = time.time()
    ###
    #cat = Catalog()
    ###
    e = time.time()
    print("Elapsed time: {}".format(e-s))
