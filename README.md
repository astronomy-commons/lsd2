# lsd2
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
$ conda create -n hipscatenv python=3.7
$ source activate hipscatenv
$ cd ~/git
$ git clone https://github.com/astronomy-commons/lsd2
$ cd lsd2
$ python -m pip install -r requirements.txt
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

The repository will look for an already partitioned catalog at that pathway `location + /output/catname`. If it doesn't find one, it will notify the user to import an existing source catalog utilizing `hipscat.Catalog.hips_import(...)`

This function requires the following parameters:
* `file_source`: this can be a directory pathway to a existing local catalog or it can be an http api-endpoing serving a list of catalog files.
* `fmt`: the catalog list file formats. Currently accepted formats are `csv`, `csv.gz`, `parquet`, and `fits`.
* `ra_kw`: the column name in the catalog files corresponding the to the sources' right ascension (RA).
* `dec_kw`: the column name in the catalog files corresponding the to the sources' declination (DEC).
* `id_kw`: the column name in the catalog files corresponding the to the sources' unique identifier (ID).
* `threshold`: the maximum number of sources each partitioned file will contain.
* `client`: a dask distributed client

Example:

```python
from dask.distributed import Client

client=Client(n_workers=12, threads_per_worker=1)

catalog.hips_import(
  file_source='/path/to/catalog/files', #or https://example.com/your_source_catalog_endpoint
  fmt='csv', #['csv', 'csv.gz', 'parquet', 'fits']
  ra_kw='ra',
  dec_kw='dec',
  id_kw='id',
  threshold=1_000_000,
  client=client
)
```

When this runs, it will create two directories in the specified `location` parameter in the catalog instantiation above: 
* `cache`: here it will save the source catalogs, and a calculated MOC (multi-order coverage) map based on the source density. From this MOC file, it creates the hierarchical partitioned structure
* `output`: here it will write the partitioned structure based on the spatial healpix ordering and source density (defined by the `threshold` parameter). The partitioned structure will follow as an example:
```bash
/Norder1/
   -> Npix5/catalog.parquet
   -> Npix15/catalog.parquet
/Norder2/
   -> Npix54/catalog.parquet
   -> Npix121/catalog.parquet
   -> Npix124/catalog.parquet
...
/NorderN/
   -> [NpixN/catalog.parquet]
```

It will also create a `meta_data.json` that it contains the basic metadata from the partitioning instatiation and running. 

### Cross-matching (returns n-matches)

Once two catalogs have been imported in this hips format, we can perform a basic prototype for cross-matching.

```python
client=Client(n_workers=12, threads_per_worker=1)

c1 = hc.Catalog('sdss', location='/path/to/hips/catalog/outputdir')
c2 = hc.Catalog('gaia', location='/path/to/hips/catalog/outputdir')
c1.cross_match(
  c2,
  n_neighbors=4, #the maximum number of nearest neighbors to locate
  dthresh=1.0.   #the distance threshold (in degrees) that limits a source cross_matching to another catalog source
  client=client  #the dask-distributed client
)
```

Returns the number of total cross-matches. 

### TODO

* Configure the output of the cross match routine.
* Consider sources that are located on the edges of neighboring pixels.
* Consider how evolving catalogs will be updated, as surveying continues.
* Consider and benchmark other parallel processing frameworks (ray, PySpark, etc).
* Enable the user to cross-match with already partitioned catalogs.
   * large data hostings, GAIA, DES, SDSS, Pan-STARRS, etc..
