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
from .partitioner import Partitioner


class Catalog:
    """An LSD2 Catalog used to import and interact with LDS2 HiPSCat formatted catalogs

    user experience:

        from hipscat import catalog as cat
        gaia = cat.Catalog('gaia')

        -> checks in source location for the 'hierarchical formatted data'

        -> if it isn't located in source notify user that the catalog must be formatted

        -> local formatting gaia.hips_import(dir='/path/to/catalog/', fmt='csv.gz') [1]
            this outputs to local directory: output/ (crib mario's code for gaia 1st 10 files)
    """

    def __init__(        self, catname="gaia", source="local", location="/epyc/projects3/sam_hipscat/"
    ):
        self.catname = catname
        self.source = source
        self.hips_metadata = None
        self.partitioner = None
        self.output_dir = None
        self.result = None
        self.location = location
        self.lsddf = None

        if self.source == "local":
            self.output_dir = os.path.join(self.location, "output", self.catname)
            if not os.path.exists(self.output_dir):
                print(
                    "No local hierarchal catalog exists, run "
                    "catalog.hips_import(file_source='/path/to/file_or_files', fmt='csv.gz')"
                )
            else:
                metadata_file = os.path.join(
                    self.output_dir, f"{self.catname}_meta.json"
                )

                if os.path.exists(metadata_file):
                    print(f"Located Partitioned Catalog: {metadata_file}")
                    with open(metadata_file) as f:
                        self.hips_metadata = json.load(f)
                else:
                    raise FileNotFoundError(
                        "Catalog not fully imported. Re-run catalog.hips_import()"
                    )

        elif self.source == "s3":
            raise NotImplementedError
        else:
            raise NotImplementedError

    def __repr__(self):
        return f"Catalog({self.catname})"

    def __str__(self):
        return f"Catalog: {self.catname}"

    def load(self, columns=None):
        """Load a catalog into a dask dataframe
        Args:
            columns: list of columns to include in the dataframe

        Returns:
            dataframe with catalog data, partitioned by HiPSCat partitioning

        """
        # dirty way to load as a dask dataframe
        if self.hips_metadata is None:
            raise ValueError(
                f"{self} hipscat metadata not found. {self}.hips_import() needs to be (re-) ran"
            )

        # validate columns input
        if not isinstance(columns, list):
            raise TypeError("columns must be supplied as a list")
        columns = columns if len(columns) > 0 else None

        # Construct path to load catalog files, using wildcards to load multiple files
        hipscat_dir = os.path.join(
            self.output_dir, "Norder*", "Npix*", "catalog.parquet"
        )

        self.lsddf = dd.read_parquet(
            hipscat_dir, calculate_divisions=True, columns=columns
        )
        return self.lsddf

    def hips_import(
        self,
        file_source="/data2/epyc/data/gaia_edr3_csv/",
        fmt="csv.gz",
        debug=False,
        verbose=True,
        limit=None,
        threshold=1_000_000,
        ra_kw="ra",
        dec_kw="dec",
        id_kw="source_id",
        client=None,
    ):
        """
        ingests a list of source catalog files and partitions them out based
        on hierarchical partitioning of size on healpix map

        supports http list of files, s3 bucket files, or local files
        formats currently supported: csv.gz, csv, fits, parquet
        """

        if "http" in file_source:
            urls = util.get_csv_urls(url=file_source, fmt=fmt)

        elif "s3" in file_source:
            raise NotImplementedError

        else:  # assume local?
            if not os.path.exists(file_source):
                raise FileNotFoundError(
                    "Local files not found at source {}".format(file_source)
                )

            # make sure local file_source is a path
            fs_clean = file_source
            if not fs_clean.endswith("/"):
                fs_clean += "/"

            urls = glob.glob("{}*{}".format(fs_clean, fmt))

        if limit:
            urls = urls[0:limit]

        if verbose:
            print(f"Attempting to format files: {len(urls)}")

        if len(urls):
            self.partitioner = Partitioner(
                catname=self.catname,
                fmt=fmt,
                urls=urls,
                id_kw=id_kw,
                order_k=10,
                verbose=verbose,
                debug=debug,
                ra_kw=ra_kw,
                dec_kw=dec_kw,
                location=self.location,
            )

            if debug:
                self.partitioner.gather_statistics()
                self.partitioner.compute_partitioning_map(
                    max_counts_per_partition=threshold
                )
                self.partitioner.write_structure_metadata()
            else:
                self.partitioner.run(client=client)
                self.__init__(self.catname, location=self.location)
        else:
            raise FileNotFoundError("No files Found!")

    def cone_search(self, ra, dec, radius, columns=None):
        """
        Perform a cone search on HiPSCat

        Args:
         ra:     float, Right Ascension
         dec:    float, Declination
         radius: float, Radius from center that circumvents, must be in degrees
         Returns:
             Dask dataframe containing the data within the cone_search
        """
        if self.hips_metadata is None:
            raise ValueError(
                f"{self} hipscat metadata not found. {self}.hips_import() needs to be (re-) ran"
            )

        if not isinstance(ra, (int, float)):
            raise TypeError(f"ra must be a number")
        if not isinstance(dec, (int, float)):
            raise TypeError(f"dec must be a number")
        if not isinstance(radius, (int, float)):
            raise TypeError(f"radius must be a number")

        # establish metadata for the returning dask.dataframe
        # user can select columns
        # returns a distance column for proximity metric

        ddf = self.load(columns=columns)
        meta = ddf._meta
        meta["_DIST"] = []

        # utilize the healpy library to find the pixels that exist
        # within the cone.
        vec = hp.ang2vec(ra, dec, lonlat=True)
        rad = np.radians(radius)
        highest_order = int(max(self.hips_metadata["hips"].keys()))
        nside = hp.order2nside(highest_order)
        pixels_to_query = hp.query_disc(nside, vec, rad, nest=True, inclusive=True)

        # query our hips metadata for partitioned pixel catalogs that
        # lie within the cone
        cone_search_map = []
        for p in pixels_to_query:
            mapped_dict = util.map_pixel_at_order(
                int(p), highest_order, self.hips_metadata["hips"]
            )
            for mo in mapped_dict:
                mapped_pixels = mapped_dict[mo]
                for mp in mapped_pixels:
                    cat_path = os.path.join(
                        self.output_dir, f"Norder{mo}", f"Npix{mp}", "catalog.parquet"
                    )
                    cone_search_map.append(cat_path)

        # only need a catalog once, remove duplicates
        # and then create the dask.dataframe that will perform
        # the map_partitions of the cone_search. The only parameter
        # we need in this dataframe is the catalog paths.
        # In the map_partitions performed later, these paths will be loaded
        # as pandas dataframes and filtered
        cone_search_map = list(set(cone_search_map))
        cone_search_map_dict = {"catalog": cone_search_map}

        nparts = len(cone_search_map)
        if nparts == 0:
            # No sources in the catalog within the disc
            return dd.from_pandas(meta, npartitions=1)

        # Create a dask_df with the paths of the partitions to be
        # loaded and filtered
        cone_search_df = dd.from_pandas(
            pd.DataFrame(
                cone_search_map_dict, columns=list(cone_search_map_dict.keys())
            ).reset_index(drop=True),
            npartitions=nparts,
        )

        # calculate the sources with the dask partitinoed ufunc
        result = cone_search_df.map_partitions(
            du.cone_search_from_daskdf,
            ra=ra,
            dec=dec,
            radius=radius,
            c_md=self.hips_metadata,
            columns=columns,
            meta=meta,
        )
        return result

    def cross_match(
        self,
        othercat=None,
        c1_cols=[],
        c2_cols=[],
        n_neighbors=1,
        dthresh=0.01,
        evaluate_margins=True,
        debug=False,
    ):
        """Perform a nearest neighbors crossmatch to another catalog
        Args:
           othercat- other hipscat catalog

           user-defined columns to return for dataframe
           c1_cols- list of [column_name]
           c2_cols- list of [column_name]
           n_neighbors - number of nearest neighbors to find for each souce in catalog1
           dthresh- distance threshold for nearest neighbors (decimal degrees)
        """

        if othercat is None:
            raise ValueError("Must specify another catalog to crossmatch with.")
        if not isinstance(othercat, Catalog):
            raise TypeError("The other catalog must be an instance of hipscat.Catalog.")
        if self.catname == othercat.catname:
            raise ValueError("Cannot cross_match catalog with self")
        if self.hips_metadata is None:
            raise ValueError(
                f"{self} hipscat metadata not found. {self}.hips_import() needs to be (re-) ran"
            )
        if othercat.hips_metadata is None:
            raise ValueError(
                f"{othercat} hipscat metadata not found. "
                f"{othercat}.hips_import() needs to be (re-) ran"
            )

        # Gather the metadata from the already partitioned catalogs
        cat1_md = self.hips_metadata
        cat2_md = othercat.hips_metadata

        # use the metadata to calculate the hipscat1 x hipscat2 map
        # this function finds the appropriate catalog parquet files to execute the crossmatches
        # returns a [
        #   [path/to/hipscat1/catalog.parquet, path/to/hipscat2/catalog.parquet]
        # ]
        hc_xmatch_map = util.map_catalog_hips(
            cat1_md["hips"], self.output_dir, cat2_md["hips"], othercat.output_dir
        )

        if debug:
            hc_xmatch_map = hc_xmatch_map[:5]
            print("DEBUGGING ONLY TAKING 5")

        # This instantiates the dask.dataframe from the hc
        #  just a table with
        #  columns = [catalog1_file_path, catalog2_file_path, other_xm_metadata...]
        matchcats_dict = util.xmatchmap_dict(hc_xmatch_map)
        # ensure the number of partitions are
        # the number cross-match operations so that memory is managed
        nparts = len(matchcats_dict[list(matchcats_dict.keys())[0]])

        matchcats_df = dd.from_pandas(
            pd.DataFrame(
                matchcats_dict, columns=list(matchcats_dict.keys())
            ).reset_index(drop=True),
            npartitions=nparts,
        )

        # establish the return columns for the returned xmatch'd dataframe's metadata
        # dask.dataframe.map_partitions() requires the metadata of the resulting
        # dataframe to be defined prior to execution. The column names are defined here
        # renamed based on catalog_name, and passed into the 'meta' variable

        # lazy load the dataframes to 'sniff' the column names. Doesn't load any data The
        # util.validate_user_input_cols function just ensures the ra, dec, and id columns aren't
        # forgotten if the user doesn't specify columns, it will grab the whole dataframe
        c1_ddf = self.load(columns=util.validate_user_input_cols(c1_cols, cat1_md))
        c2_ddf = othercat.load(columns=util.validate_user_input_cols(c2_cols, cat2_md))

        # grab the meta data for each catalogs
        c1_meta = c1_ddf._meta
        c2_meta = c2_ddf._meta

        # rename the meta-data with the catalog name prefixes
        c1_meta_prefixed = util.frame_prefix_all_cols(c1_meta, self.catname)
        c2_meta_prefixed = util.frame_prefix_all_cols(c2_meta, othercat.catname)

        # construct the dictionary that the cross_match routine utilizes to
        # - read the specified columns from the parquet file 'c1/2_cols_original'
        # - rename the columns with the prefix names for the return dataframe 'c1/2_cols_prefixed'
        # - maintain the prefixed kws for ra/dec/id so that the cross_match routine can
        #       appropriately read them from the dataframes when actually cross_matching
        all_column_dict = {
            "c1_cols_original": list(c1_meta.columns),
            "c1_cols_prefixed": list(c1_meta_prefixed.columns),
            "c2_cols_original": list(c2_meta.columns),
            "c2_cols_prefixed": list(c2_meta_prefixed.columns),
            "c1_kws_prefixed": util.catalog_prefix_kws(
                cat1_md, self.catname
            ),  # rename the ra,dec,id kws with
            "c2_kws_prefixed": util.catalog_prefix_kws(
                cat2_md, othercat.catname
            ),  # the appropriate prefixes
        }
        del c1_ddf, c2_ddf

        # TODO: Do we want to allow same catalog crossmatching, requires _2 suffix?
        # c2_cols = util.rename_meta_cols(c1_cols, c2_cols)

        # create the empty metadata dataframe along with our xmatch columns
        meta = pd.concat([c1_meta_prefixed, c2_meta_prefixed])
        meta["hips_k"] = []
        meta["hips_pix"] = []
        meta["_DIST"] = []

        # let the user know they are about to get huge dataframes
        if len(meta.columns) > 50:
            warnings.warn(
                "The number of columns in the returned dataframe is greater than 50. \
                This could potentially excede the expected computation time. \
                It is highly suggested to limit the return columns for the cross_match \
                with the c1_cols=[...], and c2_cols=[...] parameters"
            )

        # call the xmatch_from_daskdf ufunction.
        self.result = matchcats_df.map_partitions(
            du.xmatch_from_daskdf,
            all_column_dict=all_column_dict,
            n_neighbors=n_neighbors,
            dthresh=dthresh,
            evaluate_margins=evaluate_margins,
            meta=meta,
        )
        return self.result

    def visualize_sources(self, figsize=(5, 5)):
        """Visualize the sources of a catalog

        plots the high order pixel map that is
        calculated during the partitioning routine.

        Visualize from notebook

        Args:
            figsize=Tuple(x,y) for the figure size

        Returns:
             hp.mollview() of the high order pixel map that is
        calculated during the partitioning routine.
        """
        # Look for the output_dir created from
        if self.output_dir is None:
            raise FileNotFoundError(
                "hipscat output_dir not found. hips_import() needs to be (re-) ran"
            )

        outputdir_files = os.listdir(self.output_dir)
        maps = [
            x
            for x in outputdir_files
            if f"{self.catname}_order" in x and "hpmap.fits" in x
        ]

        if len(maps) == 0:
            raise FileNotFoundError(
                "map not found. hips_import() needs to be (re-) ran"
            )

        mapFn = os.path.join(self.output_dir, maps[0])
        img = hp.read_map(mapFn)
        fig = plt.figure(figsize=figsize)
        return hp.mollview(
            np.log10(img + 1),
            fig=fig,
            title=f"{self.catname}: {img.sum():,.0f} sources",
            nest=True,
        )

    def visualize_partitions(self, figsize=(5, 5)):
        """Visualize the partition structure of a catalog

        Visualize from notebook

        Args:
            figsize=Tuple(x,y) for the figure size

        Returns:
             hp.mollview() of the partitioning structure that is
        calculated during the partitioning routine.
        """
        if self.hips_metadata is None:
            raise ValueError(
                f"{self} hipscat metadata not found. {self}.hips_import() needs to be (re-) ran"
            )

        catalog_hips = self.hips_metadata["hips"]
        k = max([int(x) for x in catalog_hips.keys()])
        npix = hp.order2npix(k)
        orders = np.full(npix, -1)
        idx = np.arange(npix)
        c_orders = [int(x) for x in catalog_hips.keys()]
        c_orders.sort()

        for o in c_orders:
            k2o = 4 ** (k - o)
            pixs = catalog_hips[str(o)]
            pixk = idx.reshape(-1, k2o)[pixs].flatten()
            orders[pixk] = o

        fig = plt.figure(figsize=figsize)
        return hp.mollview(
            orders, fig=fig, max=k, title=f"{self.catname} partitions", nest=True
        )

    def visualize_cone_search(self, ra, dec, radius, figsize=(5, 5)):
        """Visualize a cone search of a catalog

        Visualize from notebook

        Args:
            ra=float, Right Ascension in decimal degrees
            dec=float, Declination in decimal degrees
            radius=float, Radius of cone in degrees
            figsize=Tuple(x,y), for the figure size
        Returns:
             hp.mollview() of a cone-search
        """
        if self.hips_metadata is None:
            raise ValueError(
                f"{self} hipscat metadata not found. {self}.hips_import() needs to be (re-) ran"
            )
        if not isinstance(ra, (int, float)):
            raise TypeError(f"ra must be a number")
        if not isinstance(dec, (int, float)):
            raise TypeError(f"dec must be a number")
        if not isinstance(radius, (int, float)):
            raise TypeError(f"radius must be a number")

        highest_order = 10
        nside = hp.order2nside(highest_order)
        vec = hp.ang2vec(ra, dec, lonlat=True)
        rad = np.radians(radius)
        pixels_to_query = hp.query_disc(nside, vec, rad, nest=True)

        # Look for the output_dir created from
        if self.output_dir is None:
            raise FileNotFoundError(
                "hipscat output_dir not found. hips_import() needs to be (re-) ran"
            )

        # look for the pixel map fits file
        outputdir_files = os.listdir(self.output_dir)
        maps = [
            x
            for x in outputdir_files
            if f"{self.catname}_order" in x and "hpmap.fits" in x
        ]
        if len(maps) == 0:
            raise FileNotFoundError(
                "map not found. hips_import() needs to be (re-) ran"
            )

        # grab the first one in the list
        map0 = maps[0]
        mapFn = os.path.join(self.output_dir, map0)

        # grab the required parameters for hp.query(disc)
        highest_order = int(mapFn.split("order")[1].split("_")[0])
        nside = hp.order2nside(highest_order)
        vec = hp.ang2vec(ra, dec, lonlat=True)
        rad = np.radians(radius)
        pixels_to_query = hp.query_disc(nside, vec, rad, nest=True)

        img = hp.read_map(mapFn)
        maxN = max(np.log10(img + 1)) + 1

        # plot pencil beam. Maybe change the maxN to a value that will
        #   never occur in np.log10(img+1).. maybe a negative number
        img[pixels_to_query] = maxN
        fig = plt.figure(figsize=figsize)
        return hp.mollview(
            np.log10(img + 1),
            fig=fig,
            title=f"Cone search of {self.catname}",
            nest=True,
        )

    def visualize_cross_match(self, othercat, figsize=(5, 5)):
        """Visualize the cross match between two catalogs

        Visualize from notebook

            Args:
                othercat=hipscat.Catalog()
                figsuze=Tuple(x,y)
            Returns:
                 hp.mollview of the overlap when crossmatching two catalogs
        """
        raise NotImplementedError("catalog.visualize_cross_match() not implemented yet")


if __name__ == "__main__":
    import time

    s = time.time()
    ###
    # cat = Catalog()
    ###
    e = time.time()
    print("Elapsed time: {}".format(e - s))
