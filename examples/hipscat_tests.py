"""Catalog instantiations that test import of several known data sets on epyc data server"""

from dask.distributed import Client
import ..hipscat as hc
import sys
sys.path.insert(0, '../')

import hipscat as hc

def test_instantiations(client=None):
    c = hc.Catalog('gaia_test')
    c.hips_import(file_source='/epyc/data/gaia_edr3_csv/', fmt='csv.gz', ra_kw='ra', dec_kw='dec',
        id_kw='source_id', debug=False, verbose=True, threshold=1_000_000, client=client,
        limit=10)

def test_instantiations2(client=None):
    c = hc.Catalog('sdss_test')
    #c.hips_import(file_source='/epyc/data/sdss_parquet/', fmt='parquet', ra_kw='RA', dec_kw='DEC',
    #    id_kw='ID', debug=False, verbose=True, threshold=1_000_000, client=client, limit=10)

def test_instantiations3(client=None):
    c = hc.Catalog('des_y1a1_gold')
    c.hips_import(file_source='https://desdr-server.ncsa.illinois.edu/despublic/y1a1_files/gold_catalogs/',
        fmt='fits', ra_kw='RA', dec_kw='DEC', id_kw='COADD_OBJECTS_ID', debug=False, verbose=False, threshold=1_000_000, client=client)

def test_instantiations4(client=None):
    '''
        ask Eric or Collin what are the contents of the ztf files
        objects or time-series
    '''
    c = hc.Catalog('ztf_dr7')
    c.hips_import(file_source='/data/epyc/projects/lsd2/pzwarehouse/ztf_dr7/', fmt='parquet', debug=True,
        ra_kw='ra', dec_kw='dec', id_kw='ps1_objid', verbose=True, threshold=250_000, limit=10, client=client)

def test_instantiations5(client=None):
    '''
        ask Eric or Collin what are the contents of the ztf files
        objects or time-series
    '''
    c = hc.Catalog('ps1')
    c.partition_from_source(file_source='/epyc/data/ps1_skinny/', fmt='csv.gz', debug=True,
        ra_kw=5, dec_kw=6, id_kw=0, verbose=True, threshold=250_000, limit=10, client=client)

if __name__ == '__main__':
    import time
    client = Client(n_workers=4, threads_per_worker=1)
    #client=None
    s = time.time()
    test_instantiations2(client=client)
    e = time.time()
    print(f'Elapsed Time: {e-s}')
