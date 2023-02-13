import numpy as np
import healpy as hp
import requests
import httplib2
import json
import os
from bs4 import BeautifulSoup, SoupStrainer


def compute_index(ra, dec, order=20):
    # the 64-bit index, viewed as a bit array, consists of two parts:
    #
    #    idx = |(pix)|(rank)|
    #
    # where pix is the healpix nest-scheme index of for given order,
    # and rank is a monotonically increasing integer for all objects
    # with the same value of pix.

    # compute the healpix pix-index of each object
    pix = hp.ang2pix(2**order, ra, dec, nest=True, lonlat=True)

    # shift to higher bits of idx
    bits=4 + 2*order
    idx = pix.astype(np.uint64) << (64-bits)

    # sort
    orig_idx = np.arange(len(idx))
    sorted_idx = np.lexsort((dec, ra, idx))
    idx, ra, dec, orig_idx = idx[sorted_idx], ra[sorted_idx], dec[sorted_idx], orig_idx[sorted_idx]

    # compute the rank for each unique value of idx (== bitshifted pix, at this point)
    # the goal: given values of idx such as:
    #   1000, 1000, 1000, 2000, 2000, 3000, 5000, 5000, 5000, 5000, ...
    # compute a unique array such as:
    #   1000, 1001, 1002, 2000, 2001, 3000, 5000, 5001, 5002, 5003, ...
    # that is for the subset of nobj objects with the same pix, add
    # to the index an range [0..nobj)
    #
    # how this works:
    # * x are the indices of the first appearance of a new pix value. In the example above,
    # it would be equal to [0, 3, 5, 6, ...]. But note that this is also the total number
    # of entries before the next unique value (e.g. 5 above means there were 5 elements in
    # idx -- 1000, 1000, 1000, 2000, 2000 -- before the third unique value of idx -- 3000)
    # * i are the indices of each unique value of idx, starting with 0 for the first one
    # in the example above, i = [0, 0, 0, 1, 1, 2, 3, 3, 3, 3]
    # * we need construct an array such as [0, 1, 2, 0, 1, 0, 0, 1, 2, 3, ...], i.e.
    # a one that resets every time the value of idx changes. If we can construct this, we
    # can add this array to idx and achieve our objective.
    # * the way to do it: start with a monotonously increasing array
    #  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, ...] and subtract an array that looks like this:
    #  [0, 0, 0, 3, 3, 5, 6, 7, 7, 7, ...]. This is an array that at each location has
    #  the index where that location's pix value appeared for the first time. It's easy
    #  to confirm that this is simply x[i].
    #
    # And this is what the following four lines implement.
    _, x, i = np.unique(idx, return_inverse=True, return_index=True)
    x = x.astype(np.uint64)
    ii = np.arange(len(i), dtype=np.uint64)
    di = ii - x[i]
    idx += di

    # remap back to the old sort order
    idx = idx[orig_idx]

    return idx


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
                              np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def map_pixel_at_order(pixel, order, _map):
    '''
    This function inputs a pixel at an order and attempts to map it
     to another map.
     
    Returns a dictionary of orders -> [pixels]
     
     pixel - (int)  - the pixel index
     order - (int)  - the healpix map order of the pixel index
     _map  - (dict) - the map you are cross matching the pixel with
                      dict[orders[pixels]]
                      e.g. _map = {"0":[0,1,2], "1":[13,14,15,16]}
                      
    This function will iterate over the orders (in reverse) in the map
     and test for 3 scenarios:
     
        1.) if the map_order == order and the pixel is within the map[order]
          pixels:
            there is a one-one mapping and we return immediately
              
        2.) if the map_order < order:
            there potentially exists map_orders' pixels at greater orders
            We bitshift >> the input pixel to the order we are iterating over
            and test if catalogs' pixels equal the shifted comparison pixel
        
        3.) if the map_order > order:
            there potentially exists map_pixels' pixels at lesser orders.
            We bitshift >> each pixel to the (map_order-order) level, and
            test if pixel we shift to at lower orders is the input pixel.
            If it is, we append the map_order pixel to the return dictionary.
    '''
    
    map_orders = [int(x) for x in _map.keys()]
    map_orders.sort(reverse=True)
    
    ret = {}
    for o in map_orders:
        #ret[o] = []
        pixs = np.asarray(_map[str(o)])
        if o == order and pixel in pixs:
            #if o is equal to order, and pixel is in the _map[order]
            # pixels. There is a one to one mapping
            ret[o] = np.array([pixel])
            return ret
            
        elif o < order:
            #if o is less than the mapping pixel order
            # we'll need to bitshift the mapping pixel to the order
            # of the catalog we are matching
            upper_pix = pixel >> 2*(order-o)
            ret[o] = pixs[pixs == upper_pix]
                
        elif o > order:
            #if o is greater than the mapping pixel order
            # we'll need to bitshift the pixels o-order levels down
            # this gives us a single pixel
            lower_pixs = pixs >> 2*(o-order)
            ret[o] = pixs[lower_pixs == pixel]
                    
    return ret


def map_catalog_hips(cat1_hips, cat1_outputdir, cat2_hips, cat2_outputdir, debug=False):
    '''
        Function to establish the xmatch mapping between two hipscat.catalog objects

        iterates over all orders in C1, then searches the second catalog hips.metadata
         for the catalog to crossmatch against. Utilizes the map_pixel_at_order() function
         to find the catalog(s) via bitshifting logic. 

         returns a list[[c1_path, c2_path]] 
    '''
    ret = []

    c1_orders = [int(x) for x in cat1_hips.keys()]
    c1_orders.sort(reverse=True)

    for o in c1_orders:
        pixs = cat1_hips[str(o)]
        for p in pixs:

            mapped_dict = map_pixel_at_order(int(p), int(o) , cat2_hips)

            for mo in mapped_dict:
                mapped_pixels = mapped_dict[mo]
                for mp in mapped_pixels:
                    c1_cat_path = os.path.join(cat1_outputdir, f'Norder{o}', f'Npix{p}',
                                               'catalog.parquet')
                    c2_cat_path = os.path.join(cat2_outputdir, f'Norder{mo}', f'Npix{mp}',
                                               'catalog.parquet')
                    res = [c1_cat_path, c2_cat_path]
                    if res not in ret:
                        ret.append([c1_cat_path, c2_cat_path])
                    
                    if debug:
                        print(f'C1 {o}:{p} --> C2 {mo}:{mp}')
    return ret


def find_pixel_neighbors_from_disk(order_k, pixel, radius=3.1):
    nside = hp.order2nside(order_k)
    npix = hp.order2npix(order_k)

    ra,dec = hp.pix2ang(nside, pixel, nest=True)
    radius = np.deg2rad(radius)
    xyz = hp.ang2vec(ra, dec)
    neighbors = hp.query_disc(nside, xyz, radius, inclusive=True, nest=True)
    return neighbors


def gnomonic(lon, lat, clon, clat):

    phi  = np.radians(lat)
    l    = np.radians(lon)
    phi1 = np.radians(clat)
    l0   = np.radians(clon)

    cosc = np.sin(phi1)*np.sin(phi) + np.cos(phi1)*np.cos(phi)*np.cos(l-l0)
    x = np.cos(phi)*np.sin(l-l0) / cosc
    y = (np.cos(phi1)*np.sin(phi) - np.sin(phi1)*np.cos(phi)*np.cos(l-l0)) / cosc

    return (np.degrees(x), np.degrees(y))


def gc_dist(lon1, lat1, lon2, lat2):
    '''
        function that calculates the distance between two points
            p1 (lon1, lat1) or (ra1, dec1)
            p2 (lon2, lat2) or (ra2, dec2)

            can be np.array()
            returns np.array()
    '''
    lon1 = np.radians(lon1); lat1 = np.radians(lat1)
    lon2 = np.radians(lon2); lat2 = np.radians(lat2)

    return np.degrees(2*np.arcsin(np.sqrt( (np.sin((lat1-lat2)*0.5))**2 + np.cos(lat1)*np.cos(lat2)*(np.sin((lon1-lon2)*0.5))**2 )))


def which_cull_and_pixorder(c1, c2):
    '''
        c1 and c2 are string pathways to two catalog.parquet files

        the logic here is that if the size of one pixel is
        greater than the other, then there is no need to retain 
        sources in the larger pixel. I.E if an order is greater, then 
        flag that catalog to be culled to the smaller order
    '''

    c1_order = int(c1.split('Norder')[1].split('/')[0])
    c1_pix = int(c1.split('Npix')[1].split('/')[0])
    c2_order = int(c2.split('Norder')[1].split('/')[0])
    c2_pix = int(c2.split('Npix')[1].split('/')[0])    

    tocull1 = False
    tocull2 = False

    if c2_order > c1_order:
        order, pix = c2_order, c2_pix
        tocull1=True
    else:
        order, pix = c1_order, c1_pix
        tocull2=True

    return order, pix, tocull1, tocull2


def xmatchmap_dict(hp_match_map):
    '''
        This instantiates the dask.dataframe dict from the hipscat xmatch map
            i.e. which catalog.parquet files will be crossmatched, along with 
            other data used in the xm
        
        returns a table with columns = [
            catalog1_file_path, 
            catalog2_file_path, 
            other_xm_metadata...
        ] 
    '''
    c1, c2 = [], []
    o, p, = [], []
    tc1, tc2 = [], []

    for m in hp_match_map:
        c1.append(m[0])
        c2.append(m[1])

        order, pix, tocull1, tocull2 = which_cull_and_pixorder(m[0], m[1])

        o.append(order)
        p.append(pix)
        tc1.append(tocull1)
        tc2.append(tocull2)

    data = {
            'C1': c1,
            'C2': c2,
            'Order' : o,
            'Pix' : p,
            'ToCull1': tc1,
            'ToCull2': tc2
    }

    return data


def validate_user_input_cols(cols, cat_md):
    '''
    if the user specifies columns in a crossmatch 
        this ensures they don't forget the ra,dec,and id
    if they don't specify columns, the read_parquet method
        will read all columns
    '''

    if len(cols):
        expected_cols = [
            cat_md['ra_kw'],
            cat_md['dec_kw'],
            cat_md['id_kw'],
        ]
        
        for ec in expected_cols:
            if ec not in cols:
                cols.append(ec)

        return cols
    return None


def frame_prefix_all_cols(df, prefix, delim='.'):
    '''
    appends a prefix to all columns in a dataframe
    '''

    cols = list(df.columns)
    rename_dict = {}
    for d in cols:
        rename_dict[d] = f'{prefix}{delim}{d}'
    ret = df.rename(rename_dict, axis=1)
    return ret


def catalog_prefix_kws(cat_md, prefix, delim='.'):
    '''
    returns prefixed kw dictionary for ra, dec, and id
     from hipscat metadata
    '''

    prefixed_kw_dict = {
        'ra_kw':f'{prefix}{delim}{cat_md["ra_kw"]}',
        'dec_kw':f'{prefix}{delim}{cat_md["dec_kw"]}', 
        'id_kw':f'{prefix}{delim}{cat_md["id_kw"]}'
    }

    return prefixed_kw_dict


def frame_cull(df, df_md, order, pix):
    '''
        df=pandas.dataframe() from catalog
        df_md=dict{}: metadata for the catalog, need the RA/DEC keywords
        order=int
        pix=int
        

        culls based on the smallest comparative order/pixel in the xmatch.
        utlizes the hp.ang2pix() to find all sources at that order,
        then only returns the dataframe containing the sources at the pixel
          -  df['hips_pix'].isin([pix])
    '''
    
    dfc = df.copy()
    dfc['hips_pix'] = hp.ang2pix(2**order, 
        dfc[df_md['ra_kw']].values, 
        dfc[df_md['dec_kw']].values, 
        lonlat=True, nest=True
    )
    dfc = dfc.loc[dfc['hips_pix'].isin([pix])]

    del df
    return dfc


def frame_gnomonic(df, df_md, clon, clat):
    '''
        method taken from lsd1:
        creates a np.array of gnomonic distances for each source in the dataframe
        from the center of the ordered pixel. These values are passed into 
        the kdtree NN query during the xmach routine.
    '''
    phi  = np.radians(df[df_md['dec_kw']].values)
    l    = np.radians(df[df_md['ra_kw']].values)
    phi1 = np.radians(clat)
    l0   = np.radians(clon)

    cosc = np.sin(phi1)*np.sin(phi) + np.cos(phi1)*np.cos(phi)*np.cos(l-l0)
    x = np.cos(phi)*np.sin(l-l0) / cosc
    y = (np.cos(phi1)*np.sin(phi) - np.sin(phi1)*np.cos(phi)*np.cos(l-l0)) / cosc

    ret = np.column_stack((np.degrees(x), np.degrees(y)))
    del phi, l, phi1, l0, cosc, x, y
    return ret


def get_csv_urls(url='https://cdn.gea.esac.esa.int/Gaia/gdr2/gaia_source/csv/', fmt='.csv.gz'):
    """
    This function parses the source url 'https://cdn.gea.esac.esa.int/Gaia/gdr2/gaia_source/csv/'
    for .csv.gz files and returns them as a list.

    :param url: the source url from where the Gaia data needs to be downloaded
    :return: list object with file names
    """

    try:
        http = httplib2.Http()
        status, response = http.request(url)
    except requests.exceptions.HTTPError as err:
        raise SystemExit(err)

    csv_files = []
    for link in BeautifulSoup(response, parse_only=SoupStrainer('a'), features="html.parser"):
        if link.has_attr('href') and link['href'].endswith(fmt):
            abs_path = url + link['href']
            csv_files.append(abs_path)

    return csv_files


if __name__ == '__main__':
    import time
    s = time.time()
    #TESTs
    e = time.time()
    print(f'Elapsed time: {e-s}')