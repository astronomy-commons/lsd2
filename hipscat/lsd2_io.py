import adlfs
import json
import os
import healpy as hp
import pandas as pd
import pyarrow.parquet as pq
import dask.dataframe as dd
from astropy.io import fits

HIPSCAT_DIR_STRUCTURE = "Norder={}/Npix={}/{}.parquet"
HIPSCAT_META_FILE_NAME = "{}_meta.json"
HIPSCAT_FITS_FILE_NAME = "{}_order10_hpmap.fits"

def _infer_file_source(pathway, storage_options=None):
    '''
    Determines whether or not the pathway is 
    from local or abstract filesystems.
    
    Current accepted protocols are:
        local pathways
        azure abfs
    '''
    _config = None
    if 'abfs' in pathway:
        source = 'azure'
        if storage_options is not None:
            try:
                _config = {
                    'account_name' : storage_options['account_name'],
                    'tenant_id'    : storage_options['tenant_id'],
                    'client_id'    : storage_options['client_id'],
                    'client_secret': storage_options['client_secret']
                }
            except:
                raise Exception("'storage_options' for azure must contain 'account_name', 'client_id', 'client_secret', and 'tenant_id' dictionary arguments")
     
    else:
        source = 'local'
               
    return source, _config


def _get_azure_fs(file_path, _config):
    '''
    Constructs the abfs object from the appropriate storage_options (_config)
    '''
    try:
        if _config is not None:
            if 'account_name' in _config.keys():
                account_name = _config['account_name']
            else:
                account_name = file_path.split('abfs://')[1].split('.dfs.core.windows.net')[0]
            tenant_id    = _config['tenant_id']
            client_id    = _config['client_id']
            client_secret= _config['client_secret']
        else:
            account_name = file_path.split('abfs://')[1].split('.dfs.core.windows.net')[0]
            tenant_id = None
            client_id = None
            client_secret = None

    except:
        raise Exception('Invalid abfs pathway. required is \'abfs://account_name.dfs.core.windows.net/hipscat/catalog\'')
    
    fs = adlfs.spec.AzureBlobFileSystem(account_name=account_name, tenant_id=tenant_id, client_id=client_id, client_secret=client_secret)
    return fs


def _map_pandas_read_parq(path, columns, engine, storage_options=None):
    '''
    since we are not using dask.read_parquet
    we are manually passing a pandas.read_parquet
    function into dask.from_map(), this alleviates 
    all of the stupidities of dask.read_parquet, and 
    is honestly faster.
    '''
    df = pd.read_parquet(
        path, 
        columns=columns, 
        engine=engine,
        storage_options=storage_options
    )
    return df


def get_norder_npix_from_catdir(pathway):
    '''
        gets order, pix from pathway like:
        '/path/to/dir/Norder=N/Dir=D/Npix=N.parquet
    '''
    if pathway[-1] == '/':
        pathway = pathway[len(pathway)-1]
    norder = int(pathway.split('Norder=')[1].split('/')[0])
    npix = int(pathway.split('Npix=')[1].split('.parquet')[0])
    return norder, npix


def get_norder_npix_from_tmpdir(pathway):
    '''
        gets order, pix from pathway like:
        '/path/to/dir/Norder=N/Npix=/catalog.parquet
    '''
    if pathway[-1] == '/':
        pathway = pathway[len(pathway)-1]
    norder = int(pathway.split('Norder=')[1].split('/')[0])
    npix = int(pathway.split('Npix=')[1].split('/')[0])
    return norder, npix


def get_hipscat_pixel_dir(norder, npix):
    '''
        returns the path to the directory
        /Norder=N/Dir=D/Npix=N.parquet format
    '''
    ndir = int(npix / 10_000) * 10_000
    return f"Norder={norder}/Dir={ndir}"


def get_hipscat_pixel_file(norder, npix):
    '''
        returns the path to the npix file under the
        /Norder=N/Dir=D/Npix=N.parquet format
    '''
    ndir = int(npix / 10_000) * 10_000
    return f"Norder={norder}/Dir={ndir}/Npix={npix}.parquet"


def read_hipsmeta_file(pathway, storage_options=None):
    '''
    Finds the hipscat json metadata file
    expects a pathway to the output hipscat catalog
    -> pathway = '/pathway/to/hipscat/catalogname'
    parses the pathway for catalog name, and uses that 
    to construct
    -> '/pathway/to/hipscat/catalogname/catalogname_meta.json'
    
    returns json.load(...)
    '''
    source, _config = _infer_file_source(pathway, storage_options)
    
    pathway = pathway if pathway[-1] != '/' else pathway[:len(pathway)-1]
    cat = pathway.split('/')[-1]
    hips_meta_file = os.path.join(pathway, HIPSCAT_META_FILE_NAME.format(cat))
    
    if not file_exists(hips_meta_file, storage_options):
        raise FileNotFoundError(f"Location HiPSCat metadata file does not exist: {hips_meta_file}")

    if source == 'local':
        with open(hips_meta_file) as f:
            return json.load(f)
        
    if source == 'azure':

        try:
            fs = _get_azure_fs(hips_meta_file, _config)
            azure_file = hips_meta_file.split('.dfs.core.windows.net/')[1]
            with fs.open(azure_file) as f:
                return json.load(f)
        except:
            raise Exception("An error occured with the adlfs download")
        

def read_fits_file(pathway, storage_options=None):
    '''
    Finds the hipscat source distribution fits file
    expects a pathway to the output hipscat catalog
    -> pathway = '/pathway/to/hipscat/catalogname'
    parses the pathway for catalog name, and uses that 
    to construct
    -> '/pathway/to/hipscat/catalogname/catalogname_order10_hpmap.fits'
    
    returns hp.read_map(...)
    '''
    source, _config = _infer_file_source(pathway, storage_options)
    
    pathway = pathway if pathway[-1] != '/' else pathway[:len(pathway)-1]
    cat = pathway.split('/')[-1]
    hips_fits_file = os.path.join(pathway, HIPSCAT_FITS_FILE_NAME.format(cat))

    if not file_exists(hips_fits_file, storage_options):
        raise FileNotFoundError(f"Location HiPSCat fits file does not exist: {hips_fits_file}")
    
    if source == 'local':
        return hp.read_map(hips_fits_file)
        
    if source == 'azure':
        #try:
        fs = _get_azure_fs(hips_fits_file, _config)
        azure_file = hips_fits_file.split('.dfs.core.windows.net/')[1]
        with fs.open(azure_file) as f:
            hdu = fits.open(f)
            return hp.read_map(hdu)
        #except:
        #    raise Exception("An error occured with the adlfs download")
            
            
def read_parquet(pathway, library='pandas', engine='pyarrow', hipsdir='catalog', columns=None, storage_options=None):
    '''
    lsd2 method to return parquet file from abstract file systems
    utilizes pandas.read_parquet for single file pathways
    or dask.from_map(pandas.read_parquet, ...) for multiple parquet files
    
    pathway: 
      can be a string for a single file
        -> returns a pandas.read_parquet() object
      can be a string for the catalog pathway
        -> returns a lazyily loaded dask dataframe object
           from constructing a list of every parquet file
           in the hipscat.
      can be a list of parquet file pathways
        -> returns a lazyily loaded dask dataframe object
           from every parquet file in the list
    
    library:
       pandas: for single parquet files
       dask: for multiple parquet files
    
    engine:
        only supporting pyarrow right now
        
    hipsdir:
        can be 'catalog' or 'neighbor'
    
    columns:
        list of columns to reduce the returned dataframe object
    '''
    if not library in ['pandas', 'dask']:
        raise Exception(f"Invalid library '{library}', only 'pandas' and 'dask' are supported")
    if not engine in ['pyarrow']:
        raise Exception(f"Invalid engine: '{engine}', only 'pyarrow' is supported")
    if not hipsdir in ['catalog', 'neighbor']:
        raise Exception(f"Invalid hipsdir: '{hipsdir}', only 'catalog' and 'neighbor' are supported")
    
    if isinstance(pathway, list):
        catalog_path = pathway[0].split(f"{hipsdir}/Norder")[0]
    else:
        catalog_path = pathway.split(f"{hipsdir}/Norder")[0]

    catalog_path = catalog_path if catalog_path[-1] != '/' else catalog_path[:len(catalog_path)-1]

    source, _config = _infer_file_source(catalog_path, storage_options)

    parquet_meta_file = read_parquet_metadata(
        os.path.join(catalog_path, hipsdir), storage_options
    )

    meta = []
    empty_table = parquet_meta_file.empty_table()
    meta = empty_table.to_pandas()
    if columns is not None:
        meta = meta[columns]

    if source == 'local':

        if library == 'pandas':
            if isinstance(pathway, list):
                raise Exception("if loading a list of parquet files, use `library=dask`")
            if not '.parquet' in pathway:
                raise Exception("pandas should only be used to load one parquet file, not a directory")
            if not file_exists(pathway):
                raise FileNotFoundError(f"pathway to parquet file: {pathway} does not exist")
            
            return pd.read_parquet(pathway, columns=columns, engine=engine)
        
        if library == 'dask':
            if isinstance(pathway, list):
                #validate every pathway or assume they will be assembled correctly?
                paths=pathway
            else:
                meta_file = read_hipsmeta_file(pathway)
                meta_hips = meta_file['hips']
                paths = []
                for k in meta_hips.keys():
                    for pixs in meta_hips[k]:
                        #glob_cat_paths = HIPSCAT_DIR_STRUCTURE.format(k, pixs, hipsdir)
                        glob_cat_paths = get_hipscat_pixel_file(k, pixs)
                        paths.append(os.path.join(pathway, hipsdir, glob_cat_paths))
                    
            return dd.from_map(
                _map_pandas_read_parq, 
                paths, 
                [columns]*len(paths), 
                [engine]*len(paths), 
                meta=meta
            )
        
    if source == 'azure':
        
        if library == 'pandas':
            if isinstance(pathway, list):
                raise Exception("if loading a list of parquet files, use `library=dask`")
            if not '.parquet' in pathway:
                raise Exception("pandas should only be used to load one parquet file, not a directory")
            if not file_exists(pathway, storage_options):
                raise FileNotFoundError(f"pathway to parquet file: {pathway} does not exist")

            if 'account_name' in _config.keys():
                _config.pop('account_name')

            return pd.read_parquet(pathway, columns=columns, engine=engine, storage_options=_config)
        
        if library == 'dask':
            if isinstance(pathway, list):
                #validate every pathway or assume they will be assembled correctly?
                paths=pathway
            else:
                meta_file = read_hipsmeta_file(pathway, storage_options)
                meta_hips = meta_file['hips']
                paths = []
                for k in meta_hips.keys():
                    for pixs in meta_hips[k]:
                        #glob_cat_paths = HIPSCAT_DIR_STRUCTURE.format(k, pixs, hipsdir)
                        glob_cat_paths = get_hipscat_pixel_file(k, pixs)
                        paths.append(os.path.join(pathway, hipsdir, glob_cat_paths))
                    
            if 'account_name' in _config.keys():
                _config.pop('account_name')

            return dd.from_map(
                _map_pandas_read_parq, 
                paths, 
                [columns]*len(paths), 
                [engine]*len(paths),
                [_config]*len(paths), 
                meta=meta
            )

      
def read_parquet_metadata(pathway, storage_options=None):
    '''
    Finds the hipscat pyarrow.parquet _metadata file
    expects a pathway to the output hipscat catalog
    -> pathway = '/pathway/to/hipscat/catalogname/subdir'
    parses the pathway for catalog name, and uses that 
    to construct
    -> '/pathway/to/hipscat/catalogname/subdir/_metadata'
    
    returns pq.read_schema(...)
    '''
    source, _config = _infer_file_source(pathway, storage_options)

    parquet_meta_file = os.path.join(pathway, '_metadata')
    
    if not file_exists(parquet_meta_file, storage_options):
        raise FileNotFoundError(f"Location HiPSCat parquet _metadata file does not exist: {parquet_meta_file}")
    
    if source == 'local':
        return pq.read_schema(parquet_meta_file)
        
    if source == 'azure':
        try:
            fs = _get_azure_fs(parquet_meta_file, _config)
            azure_file = parquet_meta_file.split('.dfs.core.windows.net/')[1]
            with fs.open(azure_file) as f:
                return pq.read_schema(f)
        except:
            raise Exception("An error occured with the adlfs download")
        
        
def file_exists(pathway, storage_options=None):
    '''
    Tests if a file exists from pathway.
        infers if file is local or from azure blob
        to call the appropriate fsspec.pathway.exists() functions

    Arguments:
    pathway: str
    '''

    source, _config = _infer_file_source(pathway, storage_options)

    if source ==  'local':
        return os.path.exists(pathway)
    
    if source == 'azure':
        fs = _get_azure_fs(pathway, _config)
        azure_file = pathway.split('.dfs.core.windows.net/')[1]
        return fs.exists(azure_file)

    raise Exception(f"Unable to infer File System from pathway: {pathway}")