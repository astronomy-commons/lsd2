import os
import json
import numpy as np
import pandas as pd
import healpy as hp
import dask.bag as db
import matplotlib.pyplot as plt
import pyarrow.parquet as pq
import pyarrow.dataset as ps

from functools import partial
from dask.distributed import Client, progress, wait

try:
    from . import util
    from . import dask_utils as du
    from . import margin_utils as mu
except ImportError:
    import util
    import dask_utils as du
    import margin_utils as mu


class Partitioner():

    def __init__(self, catname, urls=[], fmt='csv', ra_kw='ra', dec_kw='dec', 
        id_kw='source_id', order_k=10, dtypes=None, cache='cache', threshold=1_000_000,
        calculate_neighbors=True, verbose=False, debug=False, location=os.getcwd(), skiprows=None):

        self.catname = catname
        self.urls = urls
        self.skiprows = skiprows
        self.fmt = fmt
        self.ra_kw = ra_kw
        self.dec_kw = dec_kw
        self.id_kw = id_kw
        self.dtypes = dtypes
        self.location = location
        self.cache = cache
        self.order_k = order_k
        self.verbose = verbose
        self.debug = debug

        self.img = None
        self.orders = None
        self.opix = None
        self.hips_structure = None
        self.output_written = False
        self.threshold = threshold

        # the order at which we will preform the margin pixel assignments
        # needs to be equal to or greater than the highest order of partition.
        self.highest_k = order_k + 1
        self.calculate_neighbors = calculate_neighbors

        assert self.fmt in ['csv', 'parquet', 'csv.gz', 'fits'], \
            'Source catalog file format not implemented. csv, csv.gz, fits, and parquet\
             are the only currently implemented formats'

        self.set_cache_dir()
        self.set_output_dir()


    def set_cache_dir(self):
        self.cache_dir = os.path.join(self.location, self.cache, self.catname)
        if self.verbose and not os.path.exists(self.cache_dir):
            print(f'Cache Directory does not exist')
            print(f'Creating: {self.cache_dir}')
        os.makedirs(self.cache_dir, exist_ok=True)


    def set_output_dir(self):
        self.output_dir = os.path.join(self.location, self.catname)
        if self.verbose and not os.path.exists(self.output_dir):
            print(f'Output Directory does not exist')
            print(f'Creating: {self.output_dir}')
        os.makedirs(self.output_dir, exist_ok=True)


    def run(self, client=None):
        assert client is not None, 'dask.distributed.client is required'
        self.gather_statistics()
        self.compute_partitioning_map(max_counts_per_partition=self.threshold)
        self.write_partitioned_structure_wdask(client=client)
        self.structure_map_reduce_wdask(client=client)
        self.write_structure_metadata()
        self.write_parquet_metadata(hips_dir='catalog')

        if self.calculate_neighbors:
            self.write_parquet_metadata(hips_dir='neighbor')


    def gather_statistics(self, writeread_cache=True):
        # files: iterable of files to load
        # cache_dir: output director where to drop file copies (with .parquet extension)
        # k: healpix order of the counts map
        #
        # returns: img (self.order_k healpix map with object counts)

        mapFn = os.path.join(self.output_dir, f'{self.catname}_order{self.order_k}_hpmap.fits')
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
                        ra_kw=self.ra_kw, dec_kw=self.dec_kw, skiprows=self.skiprows,
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

        if self.verbose:
            print(f'Computing paritioning map')

        # the output
        orders = np.full(len(self.img), -1) # healpix map of orders
        opix = {}  # dictionary of partitions used at each level

        self.threshold = max_counts_per_partition

        neighbor_pix = []

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
            pixo, = np.nonzero(active & (imgo < self.threshold))

            neighbors_pixo, = np.nonzero(active & (imgo < self.threshold) & (imgo > 0))

            if len(pixo):
                opix[o] = pixo # store output

                for pix in neighbors_pixo:
                    margin_pix = mu.get_margin(o, pix, self.highest_k-o)
                    for mp in margin_pix:
                        neighbor_pix.append([mp, pix, o])

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
        self.neighbor_pix = pd.DataFrame(neighbor_pix, columns=['margin_pix', 'part_pix', 'part_order'])

        if self.debug:
            hp.mollview(self.orders, title=f'partitions', nest=True)
            plt.show()

            margin_img = np.zeros(hp.order2npix(self.highest_k))
            margin_img[self.neighbor_pix['margin_pix']] = 1

            hp.mollview(margin_img, title=f'margin cache', nest=True)
            plt.show()


    def write_partitioned_structure_wdask(self, client):
        if self.verbose:
            print(f'Writing partitioned structure')

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
            id_kw=self.id_kw,
            neighbor_pix=self.neighbor_pix,
            highest_k=self.highest_k,
            calculate_neighbors=self.calculate_neighbors
        )
        #r = [x.result() for x in futures]
        wait(futures)
        self.output_written = True


    def structure_map_reduce_wdask(self, client):
        if self.verbose:
            print(f'Reducing written structure')

        if not self.output_written:
            print('Error: Partitioning map must be computed before reducing')
            return

        #pix_directories = []
        #orders = list(self.opix.keys())
        #orders.sort(reverse=True)

        #for k in orders:
        #    k_dir = f'Norder{k}'
        #    npixs = self.opix[k]
        #    for pix in npixs:
        #        pix_dir = f'Npix{pix}'
        #        pix_directories.append(os.path.join(self.output_dir, k_dir, pix_dir))

        #going to have to construct this for neighbors directory as well
        cat_output_dir = os.path.join(self.output_dir, 'catalog')
        neighbor_output_dir = os.path.join(self.output_dir, 'neighbor')

        #orders = [x for x in os.listdir(self.output_dir) if 'Norder' in x]
        orders = [x for x in os.listdir(cat_output_dir) if 'Norder' in x]
        orders.sort(reverse=True)
        cat_pix_directories = []
        neigbor_pix_directories = []
        hips_structure = {}
        for k_dir in orders:
            #HACK
            #k = int(k_dir.split('Norder')[1])
            k = int(k_dir.split('Norder=')[1])
            hips_structure[k] = []

            #HACK
            #npixs = os.listdir(os.path.join(self.output_dir, k_dir))
            npixs = os.listdir(os.path.join(cat_output_dir, k_dir))
            npixs = [x for x in npixs if 'Npix' in x]
            for pix_dir in npixs:
                #HACK
                #pix = int(pix_dir.split('Npix')[1])
                pix = int(pix_dir.split('Npix=')[1])
                hips_structure[k].append(pix)

                #HACK
                #pix_directories.append(os.path.join(self.output_dir, k_dir, pix_dir))
                catpath = os.path.join(cat_output_dir, k_dir, pix_dir)
                if os.path.exists(catpath):
                    cat_pix_directories.append(catpath)

                neighbor_path = os.path.join(neighbor_output_dir, k_dir, pix_dir)
                if os.path.exists(neighbor_path):
                    neigbor_pix_directories.append(neighbor_path)

        futures1 = client.map(
            du._map_reduce, cat_pix_directories,
            filename='catalog.parquet',
            ra_kw=self.ra_kw,
            dec_kw=self.dec_kw,
            dtypes=self.dtypes
        )
        wait(futures1)

        futures2 = client.map(
            du._map_reduce, neigbor_pix_directories,
            filename='neighbor.parquet',
            ra_kw=self.ra_kw,
            dec_kw=self.dec_kw,
            dtypes=self.dtypes,
        )
        wait(futures2)

        self.hips_structure = hips_structure


    def write_structure_metadata(self):
        print('Writing hipcat structure metadata')
        if not self.hips_structure:
            print('Error in map_reduce')

        metadata = {}
        metadata['cat_name'] = self.catname
        metadata['ra_kw'] = self.ra_kw
        metadata['dec_kw'] = self.dec_kw
        metadata['id_kw'] = self.id_kw
        metadata['n_sources'] = self.img.sum()
        metadata['pix_threshold'] = self.threshold
        metadata['urls'] = self.urls
        metadata['hips'] = self.hips_structure

        dumped_metadata = json.dumps(metadata, indent=4, cls=util.NumpyEncoder)
        with open(os.path.join(self.output_dir, f'{self.catname}_meta.json'), 'w') as f:
            f.write(dumped_metadata + '\n')

        if self.debug:
            print(dumped_metadata)
    

    def write_parquet_metadata(self, hips_dir='catalog'):
        print(f'Writing parquet metadata for hipscat {hips_dir}s')
        root_dir = os.path.join(self.output_dir, hips_dir)
        tab = ps.dataset(root_dir, partitioning='hive', format='parquet')
        mc = []
        
        for f in tab.files:
            tmd = pq.read_metadata(f)
            tmd.set_file_path(f.split(f'{hips_dir}/')[1])
            mc.append(tmd)

        _meta_data_path = os.path.join(root_dir, '_metadata')
        _common_metadata_path = os.path.join(root_dir, '_common_metadata')

        pq.write_metadata(tab.schema, _meta_data_path, metadata_collector=mc)
        pq.write_metadata(tab.schema, _common_metadata_path)
            


if __name__ == '__main__':
    import time
    s = time.time()
    ## Tests
    e = time.time()
    print('Elapsed time = {}'.format(e-s))
