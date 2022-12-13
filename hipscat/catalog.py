import os
import sys
import glob
import json

from dask.distributed import Client, progress
from collections import Counter

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


    def cross_match(self, othercat=None, n_neighbors=3, dthresh=4.0, client=None, debug=False):
        '''
            before figuring out output
                output skymap histogram
        '''

        assert othercat is not None, 'Must specify another catalog to crossmatch with.'
        assert isinstance(othercat, Catalog), 'The other catalog must be an instance of hipscat.Catalog.'

        cat1_md = self.hips_metadata
        cat2_md = othercat.hips_metadata
        nmatches = 0
        hp_xmatch_map = util.map_catalog_hips(cat1_md['hips'], self.output_dir,
                cat2_md['hips'], othercat.output_dir)

        if debug:
            hp_xmatch_map = hp_xmatch_map[:5]
            print(hp_xmatch_map)
        
        if client:
            futures = client.map(
                du._cross_match,
                hp_xmatch_map,
                c1_md=cat1_md,
                c2_md=cat2_md,
                n_neighbors=n_neighbors,
                dthresh=dthresh 
            )
            progress(futures)
            
            nmatches = sum([x.result() for x in futures])

        else:
            sys.exit('Not implemented')

        print()
        print(f'Total matches {nmatches}')


    def query(self, ra, dec, radius):
        pass


if __name__ == '__main__':
    import time
    s = time.time()
    ###
    #cat = Catalog()
    ###
    e = time.time()
    print("Elapsed time: {}".format(e-s))
