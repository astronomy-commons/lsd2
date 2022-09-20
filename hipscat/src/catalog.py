import os
import sys
import glob

from dask.distributed import Client, progress

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

    def __init__(self, catname='gaia', source='local'):

        self.catname = catname
        self.source = source
        self.hierarchical_format = None
        self.partitioner = None

        if self.source == 'local':
            output_dir = os.path.join(os.getcwd(), 'output', self.catname)
            if not os.path.exists(output_dir):
                print('No local hierarchal catalog exists, run catalog.hips_import(file_source=\'/path/to/file_or_files\', fmt=\'csv.gz\')')
            else:
                print('Found partitioned catalog!')
                #TODO: implement a method to lazy load this
                self.hierarchical_format = []

        if self.source == 's3':
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
                urls = glob.glob('{}*{}'.format(file_source, fmt))
            else:
                sys.exit('Local files not found at source {}'.format(file_source))

        if limit:
            urls = urls[:limit]

        if verbose:
            print('Attempting to format files: ')
            print(urls)

        if len(urls):
            self.partitioner = pt.Partitioner(catname=self.catname, fmt=fmt, urls=urls, id_kw=id_kw,
                        order_k=10, verbose=verbose, debug=debug, ra_kw=ra_kw, dec_kw=dec_kw)

            if debug:
                self.partitioner.gather_statistics()
                self.partitioner.compute_partitioning_map(max_counts_per_partition=threshold)
            else:
                self.partitioner.run(client=client, threshold=threshold)

        else:
            print('No files Found!')


    def cross_match(self, othercat=None, dist_thresh=1.0):
        sys.exit('Not Implemented ERROR')
        assert othercat is not None, 'Must specify another catalog to crossmatch with'


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
