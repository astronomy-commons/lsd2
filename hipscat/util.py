import numpy as np
import healpy as hp
import requests
import httplib2
import json
import os
from bs4 import BeautifulSoup, SoupStrainer


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
        ret[o] = []
        pixs = _map[str(o)]
        if o == order and pixel in pixs:
            #if o is equal to order, and pixel is in the _map[order]
            # pixels. There is a one to one mapping
            ret[o].append(pixel)
            return ret
            
        elif o < order:
            #if o is less than the mapping pixel order
            # we'll need to bitshift the mapping pixel to the order
            # of the catalog we are matching
            upper_pix = pixel >> 2*(order-o)
            for p in pixs:
                if p == upper_pix:
                    ret[o].append(p)
                
        elif o > order:
            #if o is greater than the mapping pixel order
            # we'll need to bitshift the pixels o-order levels down
            # this gives us a single pixel
            for p in pixs:
                lower_pix = p >> 2*(o-order)
                if pixel == lower_pix:
                    ret[o].append(p)
                    
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
            p1 (lon1, lat1)
            p2 (lon2, lat2)

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


def catalog_columns_selector_withdtype(cat_md, cols):
    '''
        Establish the return columns for the dataframe's metadata
         dask.dataframe.map_partitions() requires the metadata of the resulting 
         dataframe to be defined prior to execution. The column names and datatypes
         are defined here and passed in the 'meta' variable

         it expects the ra_kw, dec_kw, and id_kw fields with their respective dtypes
         if the user want's other columns, it will append them

         TODO: validate user-defined cols fields {key: dtype} 
    '''
    expected_cols = {
        cat_md['ra_kw'] :'f8',
        cat_md['dec_kw']:'f8',
        cat_md['id_kw'] :'i8'
    }
    if not len(cols):
        return expected_cols
    else:
        for k in expected_cols.keys():
            if k not in cols.keys():
                cols[k] = expected_cols[k]
        return cols


def establish_pd_meta(c1_cols, c2_cols):
    colnames = []
    colnames.extend(c1_cols)
    colnames.extend(c2_cols)
    colnames.extend(['hips_k', 'hips_pix', '_DIST'])
    return colnames


def frame_cull(df, df_md, order, pix, cols=[], tocull=True):
    '''
        df=pandas.dataframe() from catalog
        df_md=dict{}: metadata for the catalog, need the RA/DEC keywords
        order=int
        pix=int
        cols=list
        tocull = bool
        
        cull the catalog dataframes based on ToCull=True/False
         and user defined columns

        culls based on the smallest comparative order/pixel in the xmatch.
        utlizes the hp.ang2pix() to find all sources at that order,
        then only returns the dataframe containing the sources at the pixel
          -  df['hips_pix'].isin([pix])
    
        TODO: select columns in the pd.read_parquet(...) command
    '''

    if tocull:
        df['hips_pix'] = hp.ang2pix(2**order, 
            df[df_md['ra_kw']].values, 
            df[df_md['dec_kw']].values, 
            lonlat=True, nest=True
        )
        df = df.loc[df['hips_pix'].isin([pix])]
    
    #user specifies which columns to return
    if len(cols):
        #ensure user doesn't cull the ra,dec,and id columns
        expected_cols = [
            df_md['ra_kw'],
            df_md['dec_kw'],
            df_md['id_kw']
        ]
        for ec in expected_cols:
            if ec not in cols:
                cols.append(ec)
        df = df[cols]

    return df


def frame_gnomonic(df, df_md, clon, clat):
    '''
    method taken from lsd1
        creates a list of gnomonic distances for each source in the dataframe
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