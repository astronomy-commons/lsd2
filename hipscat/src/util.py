import numpy as np
import healpy as hp
import requests
import httplib2
from bs4 import BeautifulSoup, SoupStrainer

def find_pixel_neighbors_from_disk(order_k, pixel, radius=3.1):
    nside = hp.order2nside(order_k)
    npix = hp.order2npix(order_k)

    ra,dec = hp.pix2ang(nside, pixel, nest=True)
    radius = np.deg2rad(radius)
    xyz = hp.ang2vec(ra, dec)
    neighbors = hp.query_disc(nside, xyz, radius, inclusive=True, nest=True)
    return neighbors

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
