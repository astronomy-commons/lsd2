import os
import sys
import glob
import json

import numpy as np
from dask.distributed import Client, progress
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


    #lazyloading
    def __load(self, Norder=None, Npix=None):
        sys.exit('Not Implemented ERROR')
        #implement lazy loading here


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
            urls = urls[:limit]

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


    def distributued_cross_match(self, othercat=None, c1_cols=[], c2_cols=[], n_neighbors=1, dthresh=0.01, client=None, debug=False):
        '''
            Deprecated:
            Utilizes dask.distributed to map the crossmatch algorithm across
            the hipscat x hipscat map.
        '''

        assert othercat is not None, 'Must specify another catalog to crossmatch with.'
        assert isinstance(othercat, Catalog), 'The other catalog must be an instance of hipscat.Catalog.'

        cat1_md = self.hips_metadata
        cat2_md = othercat.hips_metadata

        hp_xmatch_map = util.map_catalog_hips(cat1_md['hips'], self.output_dir,
                cat2_md['hips'], othercat.output_dir)

        print(len(hp_xmatch_map))

        if debug:
            hp_xmatch_map = hp_xmatch_map[:5]
            print('DEBUGGING ONLY TAKING 5')

        if client:
            futures = client.map(
                du._cross_match2,
                hp_xmatch_map,
                c1_md=cat1_md,
                c2_md=cat2_md,
                c1_cols=c1_cols,
                c2_cols=c2_cols,
                n_neighbors=n_neighbors,
                dthresh=dthresh
            )
            progress(futures)

            self.result = dd.concat([x.result() for x in futures])

        else:
            sys.exit('Not implemented')

        return self.result


    def cross_match(self, othercat=None, c1_cols=[], c2_cols=[], n_neighbors=1, dthresh=0.01, debug=False):
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
            print(len(hc_xmatch_map))
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

        #estanblish the return columns for the returned dataframe's metadata
        # dask.dataframe.map_partitions() requires the metadata of the resulting
        # dataframe to be defined prior to execution. The column names and datatypes
        # are defined here and passed in the 'meta' variable
        c1_cols = util.catalog_columns_selector_withdtype(cat1_md, c1_cols)
        c2_cols = util.catalog_columns_selector_withdtype(cat2_md, c2_cols)

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
            c1_cols.keys(), c2_cols.keys(),
            n_neighbors=n_neighbors,
            dthresh=dthresh,
            meta = meta
        )
        return self.result


if __name__ == '__main__':
    import time
    s = time.time()
    ###
    #cat = Catalog()
    ###
    e = time.time()
    print("Elapsed time: {}".format(e-s))
