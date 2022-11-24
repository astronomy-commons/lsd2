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


def xmatch_distributed(client=None):
    import numpy as np

    c1 = hc.Catalog('des_y1a1_gold')
    c2 = hc.Catalog('gaia_real')

    #define columns to retain for each catalog after cross_match
    c1_cols = ['RA', 'DEC', 'COADD_OBJECTS_ID',]
    c2_cols = ['ra', 'dec', 'pmra', 'pmdec', 'source_id']

    result = c1.futures_cross_match(
        c2, 
        c1_cols=c1_cols, 
        c2_cols=c2_cols, 
        n_neighbors=1, 
        client=client, 
        debug=True
    ).assign(
        filter1=lambda x: x.COADD_OBJECTS_ID % 5, 
       	filter2=lambda x:np.sqrt(x.pmra**2 + x.pmdec**2)
    ).query(
        'filter1 > 3 and filter2 > 10.0'
    ).compute()

    print(len(result))

def xmatch_dataframe():
    import numpy as np

    c1 = hc.Catalog('des_y1a1_gold')
    c2 = hc.Catalog('gaia_real')

    #define columns to retain for each catalog after cross_match

    # test to see if datatype is necessary
    # look into opening entire catalog into one dataframe with pathway=glob
    c1_cols = {}
    c2_cols = {'pmra':'f8', 'pmdec':'f8'}


    result = c1.cross_match(
        c2, 
        c1_cols=c1_cols, 
        c2_cols=c2_cols, 
        n_neighbors=1, 
        debug=True
    ).assign(
        filter1=lambda x: x.COADD_OBJECTS_ID % 5, 
       	filter2=lambda x: np.sqrt(x.pmra**2 + x.pmdec**2)
    ).query(
        'filter1 > 3 and filter2 > 10.0'
    ).compute()

    print(result.head())
    print(len(result))

if __name__ == '__main__':
    import time
    import gc
    #gc.set_debug(gc.DEBUG_LEAK)
    client = Client(local_directory='/epyc/projects3/sam_hipscat/', n_workers=48, threads_per_worker=2)
    #client=None
    s = time.time()
    #download_sdss(client=client)
    xmatch_dataframe()
    e = time.time()
    print(f'Elapsed Time: {e-s}')
    #client.close()
