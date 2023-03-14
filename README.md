
<img src="https://user-images.githubusercontent.com/113376043/203642349-4ff18331-6fb4-40da-80dd-b29e234f56c3.png" style="width: 250px"/>

# LSD 2

Large Survey Database, v2

This repository is designed to facilitate and enable spatial functionality for extremely large astronomical databases (i.e. querying and crossmatching O(1B) sources). 
It is designed to function in parallel frameworks, the current prototype utilizes the [dask distributed](https://distributed.dask.org/en/stable/) framework. The strength of this package relies on spatially partioning the database while considering source density as well. This will enable same-sized file comparisons in cross-matching.

## Install

### Setting up your machine's environment

Start with a conda python environment:

```bash
$ cd ~
$ wget https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh
$ bash Anaconda3-2022.05-Linux-x86_64.sh
$ source .bashrc
$ conda update -n -base -c defaults conda
```

Note that you *may* already have conda and python installed. Future releases will be available through pip install, but current prototype involves git-cloning and setup installing.

```bash
$ cd ~
$ mkdir git
```

### Setting up virtual development environment

```bash
$ conda create -n hipscatenv python=3.10
$ source activate hipscatenv
$ cd ~/git
$ git clone https://github.com/astronomy-commons/lsd2
$ cd lsd2
$ python -m pip install -r requirements.txt
$ ipython kernel install --user --name=hipscatenv
$ source setup.bash
$ pip install -e
```

### Running unit tests

```bash
$ cd ~/git/lsd2/tests
$ python -m unittest
```

This is a great way to confirm that your development environment is properly configured and you're ready to start working on the code.

## Current en-Prototype Usage

### Partitioning your catalog

In order to get to the step of actually calculating a cross-match, the source catalogs will need to be partitioned in a way to enable efficient/scalable cross-matches. The way that we are planning to partition the catalogs is through dynamically indexing the sources in healpix space based on catalog density per index. We foresee that parallel processed cross-matching will be further facilitated by this indexing schema. 

To instantiate a catalog:

```python
import hipscat as hc

catalog = hc.Catalog('gaia', location='/path/to/hips/catalog/outputdir')
```

The repository will look for an already partitioned catalog at that pathway `location + /output/catname`. Currently the `location` keyword supports finding hipscat objects on
* local disk
* azure blob file system (abfs)
* aws s3 bucket (s3 boto) coming soon!

If it doesn't find one, it will notify the user to import an existing source catalog utilizing the `hipscat.Partitioner` class.

The instantiation requires the following parameters:
* `urls`: this is a list of string pathways to a existing local catalog or it can be an http api-endpoing serving a list of catalog files. You can utilize the `hipscat.util.get_cat_urls(...)` method to create these.
* `fmt`: the catalog list file formats. Currently accepted formats are `csv`, `csv.gz`, `parquet`, and `fits`.
* `ra_kw`: the column name in the catalog files corresponding the to the sources' right ascension (RA).
* `dec_kw`: the column name in the catalog files corresponding the to the sources' declination (DEC).
* `id_kw`: the column name in the catalog files corresponding the to the sources' unique identifier (ID).
* `threshold`: the maximum number of sources each partitioned file will contain.
* `client`: a dask distributed client

Example:

```python
import numpy as np
import hipscat as hc
from dask.distributed import Client

#specify the dask distributed client
client = Client(n_workers=12, threads_per_worker=1)

#specify the location of the catalog you wish to partition
# and grab the urls
file_source='http://cdn.gea.esac.esa.int/Gaia/gdr3/gaia_source/'
fmt = 'csv.gz' # can be csv, csv.gz, parquet, or fits 
urls = hc.util.get_cat_urls(url=file_source, fmt=fmt)

#some parquet dtypes aren't always interpreted correctly
# if you want to specify specific dtypes, you can do it with this dictionary
manual_dtype = {'libname_gspphot':'unicode'}

#instantiate the lsd2 partitioner object
partitioner = hc.Partitioner(
    catname='gaia', 
    fmt=fmt, 
    urls=urls, 
    id_kw='source_id',
    ra_kw='ra',
    dec_kw='dec',
    threshold=1_000_000,
    skiprows=np.arange(0,1000), # skips the first 1000 rows of gaia dr3 column metadata
    dtypes=manual_dtype
)

partitioner.run(client=client)
```

When this runs, it will create directories in the specified `location` parameter in the catalog instantiation above: 
* `cache`: here it will save the source catalogs for faster partitioning if the process needs to be re-ran.
* `catname/catalog`: here is where it writes the partitioned structure based on the spatial healpix ordering and source density (defined by the `threshold` parameter) along with neighbor margin sources for accurate cross-matches between hipscats. The partitioned structure will follow as an example:
```bash
_metadata
_common_metadata
/Norder1/
   -> Npix5/
        -> catalog.parquet
   -> Npix15/
        -> catalog.parquet
/Norder2/
   -> Npix54/
        -> catalog.parquet
   -> Npix121/
        -> catalog.parquet
   -> Npix124/
        -> catalog.parquet
...
/NorderN/
   -> [NpixN/
        -> catalog.parquet
   ]
```
* `catname/neighbor`: here is where it writes the partitioned structure for the calculated margins for each of the HiPSCat directories above. Each directory will contain a margin area surrounding each pixel kept in this hipscat directory.
```bash
_metadata
_common_metadata
/NorderN/
   -> [NpixN/
        -> neighbor.parquet
   ]
```

It will also create two files:
* a catalog `metadata.json` that contains the basic metadata from the partitioning instatiation and running.
* a healpix map that saves the spatial source distribution at a high order. 

## Example Usage
A full tutorial of use-cases are viewable in the `/examples/example_usage.ipynb` notebook.

### Cone-searching (return a comput-able `dask.dataframe`)

Perform cone-search on hipscat. Input params:
* ra - Right Ascension (decimal degrees)
* dec - Declination (decimal degrees)
* radius - Radius of disc to search in (decimal degrees)
* columns - List of columns in catalog to return.

Returns: `dask.dataframe`. 
* can leverage the dataframe api and further analyze result

```python
c1 = hc.Catalog('gaia', location='/path/to/hips/catalog/')
mygaia_conesearch = c1.cone_search(
  ra=56, 
  dec=20,
  radius=10.0,
  columns=['ra', 'dec', 'source_id', 'pmra', 'pmdec']
)

#Nothing is executed until you performe the .compute()
mygaia_conesearch.compute()
```


### Cross-matching (returns comput-able `dask.dataframe`)

Once two catalogs have been imported in this hips format, we can perform a basic prototype for cross-matching (nearest neighbors). The user can select the columns they want from each catalog that will be in the resulting cross-matched dataframe, which will speed up the result and is highly suggested. If the user doesn't specify any columns, it will grab all of them and can increase computation time significantly. 

```python
client=Client(n_workers=12, threads_per_worker=1)

c1 = hc.Catalog('sdss', location='/path/to/hips/catalog/outputdir')
c2 = hc.Catalog(
  'gaia', 
  location='abfs://sdss.dfs.core.windows.net/hipscat/', 
  storage_options={
    'account_name' : '...', 
    'tenant_id'    : '...',
    'client_id'    : '...',
    'client_secret': '...'
  }
)

c1_cols = []
c2_cols = ['pmdec', 'pmra']

result = c1.cross_match(
  c2,
  c1_cols=c1_cols,
  c2_cols=c2_cols,
  n_neighbors=4, #the maximum number of nearest neighbors to locate
  dthresh=1.0.   #the distance threshold (in degrees) that limits a source cross_matching to another catalog source
)
```

Returns a `dask.dataframe` of the result where all columns selected are prefixed by `catalogname.`. From this result the user can utilize the `dask.dataframe` api, and leverage its strengths for example:

```python
r = result.compute() # performs the cross match computation

r2 = result.assign( #create a new column from the result
  pm=lambda x: np.sqrt(x['gaia.pmra']**2 + x['gaia.pmdec']**2)
).compute()

r3 = result.assign( #create a new column from the result
  pm=lambda x: np.sqrt(x['gaia.pmra']**2 + x['gaia.pmdec']**2)
).query( #filter the result 
  'pm > 1.0' #prefixed column names must be surrounded by backticks. i.e `gaia.pmdec` > 10
).to_parquet( #write the result to a parquet file
  "path/to/my/parquet/"
).compute() 
```
