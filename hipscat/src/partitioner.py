import os
import glob
import random
import numpy as np
import pandas as pd
import healpy as hp
import dask.bag as db
import matplotlib.pyplot as plt

from functools import partial
from dask.distributed import Client, progress

from . import util
from . import dask_utils as du
#import util
#import dask_utils as du


class Partitioner():
    
    def __init__(self, catname, urls=[], fmt='csv', ra_kw='ra', dec_kw='dec', 
        id_kw='source_id',  order_k=5, output='output', cache='cache', 
        verbose=False, debug=False):
        
        self.catname = catname
        self.urls = urls
        self.fmt = fmt
        self.ra_kw = ra_kw
        self.dec_kw = dec_kw
        self.id_kw = id_kw
        self.cache = cache
        self.output = output
        self.order_k = order_k
        self.verbose = verbose
        self.debug = debug
        self.img = None
        self.orders = None
        self.opix = None
        self.output_written = False

        assert self.fmt in ['csv', 'parquet', 'csv.gz', 'fits'], \
            'Source catalog file format not implemented. csv, csv.gz, fits, and parquet\
             are the only currently implemented formats'

        self.set_cache_dir()
        self.set_output_dir()
    

    def set_cache_dir(self):
        self.cache_dir = os.path.join(os.getcwd(), self.cache, self.catname)
        if self.verbose and not os.path.exists(self.cache_dir):
            print(f'Cache Directory does not exist')
            print(f'Creating: {self.cache_dir}')
        os.makedirs(self.cache_dir, exist_ok=True)


    def set_output_dir(self):
        self.output_dir = os.path.join(os.getcwd(), self.output, self.catname)
        if self.verbose and not os.path.exists(self.output_dir):
            print(f'Output Directory does not exist')
            print(f'Creating: {self.output_dir}')
        os.makedirs(self.output_dir, exist_ok=True)


    def run(self, client=None, threshold=1_000_000):
        self.gather_statistics()
        self.compute_partitioning_map(max_counts_per_partition=threshold)
        if client:
            self.write_partitioned_structure_wdask(client=client)
            self.structure_map_reduce_wdask(client=client)
        else:
            self.write_partitioned_structure()
            self.structure_map_reduce()


    def gather_statistics(self, writeread_cache=True):
        # files: iterable of files to load
        # cache_dir: output director where to drop file copies (with .parquet extension)
        # k: healpix order of the counts map
        #
        # returns: img (self.order_k healpix map with object counts)
        
        mapFn = os.path.join(self.cache_dir, f'{self.catname}_order{self.order_k}_hpmap.fits')
        if writeread_cache:
            if os.path.exists(mapFn):
                try:
                    self.img = hp.read_map(mapFn)
                except:
                    print('Warning. Error reading cached map. Attempting to re-calculate it')
                    self.img = None
        
        if self.img is None:   
            if self.verbose:
                print(f'Caching input source catalogs into parquet files: {len(self.urls)} files')

            self.img = db.from_sequence(
                self.urls,
                partition_size=1
            ).reduction( 
                partial(
                    du._gather_statistics_hpix_hist, 
                        k=self.order_k, cache_dir=self.cache_dir, fmt=self.fmt,
                        ra_kw=self.ra_kw, dec_kw=self.dec_kw
                    ), 
                sum, split_every=3
            ).compute()

            if writeread_cache:
                if self.verbose:
                    print(f'Saving source healpix map.')
                hp.write_map(mapFn, self.img)        
        
        if self.debug:
            hp.mollview(np.log10(self.img+1), title=f'{self.img.sum():,.0f} sources', nest=True)
            plt.show()


    def compute_partitioning_map(self, max_counts_per_partition=1_000_000):

        # the output
        orders = np.full(len(self.img), -1) # healpix map of orders
        opix = {}  # dictionary of partitions used at each level

        thresh = max_counts_per_partition

        # Top-down partitioning. Given a dataset partitioned at order k
        # bin it to higher orders (starting at 0, and working our way
        # down to k), and at each order find pixels whose count has
        # fallen below the threshold 'thresh' and record them to be
        # stored at this order.
        #
        # Outputs: opix: dict of order => pixel IDs
        #          orders: a k-order array storing the order at which this k-order pixel should be stored.
        #
        # There's a lot of fun numpy/healpix magic down below, but it all boils
        # down to two things:
        #
        # * In healpix NEST indexing scheme, when the order of the pixelization
        #   is raised by 1, each pixel is subdivided into four subpixels with
        #   pixel IDs [4*idx_o, 4*idx+1, 4*idx+2, 4*idx+3]. That means that if
        #   you need to find out in which _higher_ order pixel some pixel ID
        #   is, just integer-divide it by 4**(k-o) where k is your current order
        #   and o is the higher order. Example: pixel 49 at order 3 fall within
        #   pixel 12 at order 2, 3 at order 1, and 0 at order 0. Easy!
        # * How do you efficiently bin pixels _values_ to a higher order? To go
        #   one order up, you need to sum up groups of 4 pixels in the array
        #   (we're "reversing" the subdivision). If we go up by two orders, it's
        #   groups of 4*4... generally, it's 4**(k-o). This summing can be done
        #   very efficiently with a bit of numpy trickery: reshape the 1-D healpix
        #   array to a 2-d array where the 2nd dimension is equal to 4**(k-o),
        #   and then simply sum along that axis. The result leaves you with the
        #   array rebinned to level o.
        #
        k = hp.npix2order(len(self.img))
        idx = np.arange(len(self.img))
        for o in range(0, k+1):
            # the number of order-k pixels that are in one order-o pixel.
            # integer-dividing order-k pixel index (NEST scheme) with 
            # this value will return the order-o index it falls within.
            k2o = 4**(k-o)

            # order o-sized bool mask listing pixels that haven't yet been
            # assigned a partition.
            active = (orders == -1).reshape(-1, k2o).any(axis=1)

            # rebin the image to order o
            imgo = self.img.reshape(-1, k2o).sum(axis=1)

            # find order o indices where pixel counts are below threshold.
            # These are the one which we will keep at this order.
            pixo, = np.nonzero(active & (imgo < thresh))

            if len(pixo):
                opix[o] = pixo # store output

                # record the order-k indices which have been assigned to the
                # partition at this level (order o). This makes it easy to
                # check which ones are still left to process (see the 'active=...' line above)
                pixk = idx.reshape(-1, k2o)[pixo].flatten()  # this bit of magic generates all order-k 
                                                            # indices of pixels that fall into order-o
                                                            # pixels stored in pixo
                orders[pixk] = o
                if self.verbose:
                    print(o, np.count_nonzero(orders == -1), len(pixo))
                 

        assert not (orders == -1).any()
        self.orders = orders
        self.opix = opix

        if self.debug:
            hp.mollview(self.orders, title=f'partitions', nest=True)
            plt.show()


    def write_partitioned_structure(self):
        if not self.opix:
            print('Error: Partitioning map must be computed before writing partitioned structure')
            return

        #need to go from the top down:
        order_of_orders = list(self.opix.keys())
        order_of_orders.sort(reverse=True)

        #for order in order_of_orders:
        #    print(order)
        #    order_partition_fd = os.path.join(dir, f'Norder{order}')
        #    if not os.path.exists(order_partition_fd):
        #        os.makedirs(order_partition_fd)
        #    
        #    for pix in self.opix[order]:
        #        partition_pix_fd = os.path.join(order_partition_fd, f'Npix{pix}')
        #        if not os.path.exists(partition_pix_fd):
        #            os.makedirs(partition_pix_fd)

        #TODO block needs to have the ability to be parallelized

        urls = self.urls
        total_sources = self.img.sum()
        summary = 0

        audit_counts = {}
        for k in order_of_orders:
            audit_counts[k] = []

        if self.debug:
            urls = urls[:10]

        for url in urls:
            base_filename = os.path.basename(url).split('.')[0]
            parqFn = os.path.join(self.cache_dir, base_filename + '.parquet')
            df = pd.read_parquet(parqFn, engine='pyarrow')

            print(f'Finding sources in file: {base_filename}')
            for k in order_of_orders:
                print(f'Order level {k}')

                df['hips_k'] = k
                df['hips_pix'] = hp.ang2pix(2**k, 
                                            df[self.ra_kw].values, 
                                            df[self.dec_kw].values, 
                                            lonlat=True, 
                                            nest=True
                                            )

                if self.verbose:
                    print(f'Partition source npix for order {k}')
                    print(self.opix[k])
                    print(f'Catalog source npix for order {k}')
                    print(np.unique(df['hips_pix']))

                order_df = df.loc[df['hips_pix'].isin(self.opix[k])]
                if self.debug:
                    non_order_df = df.loc[~df['hips_pix'].isin(self.opix[k])]
                    debug_mask = np.full(hp.order2npix(k), 0)
                    debug_mask[self.opix[k]]=1
                    debug_mask[np.unique(non_order_df['hips_pix'])]=2
                    debug_mask[np.unique(order_df['hips_pix'])]=3
                    #hp.mollview(debug_mask, nest=True)
                    #plt.show()
                #order_mask = np.in1d(df['hips_pix'].values, self.opix[k])
                #order_df = df[order_mask]

                if self.verbose:
                    print(f'Found n sources {len(order_df)}')
                
                audit_counts[k].append(len(order_df))

                #reset the df so that it doesn't include the already partitioned sources
                # ~df['column_name'].isin(list) -> sources not in order_df sources
                if self.verbose:
                    print(f'Number of sources in original DF before culling based off order_k sources: {len(df)}')

                df = df.loc[~df[self.id_kw].isin(order_df[self.id_kw])]
                if self.verbose:
                    print(f'Number of sources after culling: {len(df)}')

                #groups the sources in order_k pixels, then outputs them to the base_filename sources
                ret = order_df.groupby(['hips_k', 'hips_pix']).apply(
                        du._to_hips, 
                        hipsPath=self.output_dir, 
                        base_filename=base_filename
                )
                
                if self.verbose:
                    print()

                if len(df) == 0:
                    break

        imported_sources = 0
        for k in order_of_orders:
            imported_sources += sum(audit_counts[k])

        if self.verbose:
            print(f'Expected sources: {total_sources}')
            print(f'Imported sources: {imported_sources}')

        self.output_written = True
        print("Finished making File partitioned structure")


    def structure_map_reduce(self):
        if not self.output_written:
            print('Error: Partitioning map must be computed before reducing')
            return

        '''
            TODO: Must be written for parrellizaiton (sp?)
        '''

        orders = os.listdir(self.output_dir)
        orders.sort(reverse=True )
        for k in orders:
            npixs = os.listdir(os.path.join(self.output_dir, k))
            print(f'Looking in directory: {os.path.join(self.output_dir, k)}')
            for pix in npixs:
                files = os.listdir(os.path.join(self.output_dir, k, pix))
                dfs = []

                if len(files) == 1:
                    fn = os.path.join(self.output_dir, k, pix, files[0])
                    df = pd.read_parquet(fn, engine='pyarrow')
                    new_fn = os.path.join(self.output_dir, k, pix, 'catalog.parquet')
                    if self.verbose:
                        print(f'Renaming {files[0]} to catalog.parquet')
                    os.rename(fn, new_fn)
                elif len(files):
                    if self.verbose:
                        print(f'Concatenating {len(files)} files into Norder{k}/Npix{pix} catalog.parquet')
                    for f in files:
                        fn = os.path.join(self.output_dir, k, pix, f)
                        dfs.append(pd.read_parquet(fn, engine='pyarrow'))
                        os.remove(fn)

                    df = pd.concat(dfs, sort=False)
                    output_fn = os.path.join(self.output_dir, k, pix, 'catalog.parquet')
                    df.to_parquet(output_fn)

                if self.verbose:
                    #lengths should be around the max_counts_per_partition
                    print(f'Length of dataframe for order {k}, npix {pix}: {len(df)}')


    def write_partitioned_structure_wdask(self, client):
        if not self.opix:
            print('Error: Partitioning map must be computed before writing partitioned structure')
            return

        #need to go from the top down:
        orders = list(self.opix.keys())
        orders.sort(reverse=True)

        urls = self.urls
        total_sources = self.img.sum()
        summary = 0

        audit_counts = {}
        for k in orders:
            audit_counts[k] = []

        pfiles = []

        for url in urls:
            base_filename = os.path.basename(url).split('.')[0]
            pfiles.append(os.path.join(self.cache_dir, base_filename + '.parquet'))

        futures = client.map(
            du._write_partition_structure, pfiles, 
            cache_dir=self.cache_dir,
            output_dir=self.output_dir,
            orders=orders,
            opix=self.opix,
            ra_kw=self.ra_kw,
            dec_kw=self.dec_kw,
            id_kw=self.id_kw
        )

        progress(futures)
        self.output_written = True


    def structure_map_reduce_wdask(self, client):
        if not self.output_written:
            print('Error: Partitioning map must be computed before reducing')
            return

        orders = os.listdir(self.output_dir)
        orders.sort(reverse=True)
        pix_directories = []
        for k in orders:
            npixs = os.listdir(os.path.join(self.output_dir, k))
            for pix in npixs:
                pix_directories.append(os.path.join(self.output_dir, k, pix))
        
        futures = client.map(du._map_reduce, pix_directories)
        progress(futures)


if __name__ == '__main__':
    import time
    s = time.time()
    ###
    #urls = glob.glob('/epyc/data/gaia_edr3_csv/*.csv.gz')[:10]
    
    urls = glob.glob('/epyc/data/sdss_parquet/*.parquet')[:10]
    imp = Partitioner(catname='sdss', urls=urls, order_k=10, verbose=True, debug=True)
    imp.gather_statistics()
    hp.mollview(np.log10(imp.img+1), title=f'{imp.img.sum():,.0f} sources', nest=True)
    plt.show()
    imp.compute_partitioning_map(max_counts_per_partition=1_000_000)
    hp.mollview(imp.orders, title=f'partitions', nest=True)
    plt.show() 

    #paraleleleize these functions!
    #imp.write_partitioned_structure()
    #imp.structure_map_reduce()
    ###
    e = time.time()
    print('Elapsed time = {}'.format(e-s))