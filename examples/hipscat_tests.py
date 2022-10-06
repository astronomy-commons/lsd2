"""Catalog instantiations that test import of several known data sets on epyc data server"""

import sys
sys.path.insert(0, '../')
import hipscat as hc
from dask.distributed import Client


def download_gaia(client=None):
    c = hc.Catalog('gaia_real', location='/epyc/projects3/sam_hipscat/')
    c.hips_import(file_source='/epyc/data/gaia_edr3_csv/', fmt='csv.gz', ra_kw='ra', dec_kw='dec',
        id_kw='source_id', debug=False, verbose=True, threshold=1_000_000, client=client)

def download_sdss(client=None):
    c = hc.Catalog('sdss_test', location='/epyc/projects3/sam_hipscat/')
    c.hips_import(file_source='/epyc/data/sdss_parquet/', fmt='parquet',
        ra_kw='RA', dec_kw='DEC', id_kw='ID', debug=False, verbose=True, threshold=1_000_000, 
        limit=5, client=client)

def download_des(client=None):
    c = hc.Catalog('des_y1a1_gold', location='/epyc/projects3/sam_hipscat/')
    c.hips_import(file_source='https://desdr-server.ncsa.illinois.edu/despublic/y1a1_files/gold_catalogs/',
        fmt='fits', ra_kw='RA', dec_kw='DEC', id_kw='COADD_OBJECTS_ID', 
        debug=True, verbose=True, threshold=1_000_000, client=client)

def download_ztf(client=None):
    '''
        ask Eric or Collin what are the contents of the ztf files
        objects or time-series
    '''
    c = hc.Catalog('ztf_dr7', location='/epyc/projects3/sam_hipscat/')
    c.hips_import(file_source='/data/epyc/projects/lsd2/pzwarehouse/ztf_dr7/', fmt='parquet', debug=True,
        ra_kw='ra', dec_kw='dec', id_kw='ps1_objid', verbose=True, threshold=250_000, limit=10, client=client)

def download_ps1(client=None):
    '''
        ask Eric or Collin what are the contents of the ztf files
        objects or time-series
    '''
    c = hc.Catalog('ps1', location='/epyc/projects3/sam_hipscat/')
    c.partition_from_source(file_source='/epyc/data/ps1_skinny/', fmt='csv.gz', debug=True,
        ra_kw=5, dec_kw=6, id_kw=0, verbose=True, threshold=250_000, limit=10, client=client)


def xmatch(client=None):
    c1 = hc.Catalog('sdss_test')
    c2 = hc.Catalog('gaia_real')
    c1.cross_match(c2, client=client, debug=True)

if __name__ == '__main__':
    import time
    client = Client(local_directory='/epyc/projects3/sam_hipscat/', n_workers=48, threads_per_worker=1)
    #client=None
    s = time.time()
    #download_sdss(client=client)
    xmatch(client=client)
    e = time.time()
    print(f'Elapsed Time: {e-s}')
