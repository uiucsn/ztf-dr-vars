import logging
import os
import shutil
from contextlib import closing
from tempfile import TemporaryDirectory
from urllib.request import urlopen

from astropy.io import ascii

from ch_vars.common import *


def download_from_ftp(url, directory):
    filename = os.path.basename(url)
    # CDS read driver requies filename to be consistent with ReadMe
    if filename.endswith('.gz'):
        filename, _ext = os.path.splitext(filename)
    path = os.path.join(directory, filename)
    logging.info(f'Downloading {url} to {path}')
    with closing(urlopen(url)) as r, open(path, 'wb') as fh:
        shutil.copyfileobj(r, fh)
    return path


def vsx_dat(directory):
    url = 'ftp://cdsarc.u-strasbg.fr/pub/cats/B/vsx/vsx.dat.gz'
    return download_from_ftp(url, directory)


def readme(directory):
    url = 'ftp://cdsarc.u-strasbg.fr/pub/cats/B/vsx/ReadMe'
    return download_from_ftp(url, directory)


def get_data():
    with TemporaryDirectory() as directory:
        logging.info('Reading VSX table')
        table = ascii.read(vsx_dat(directory), readme=readme(directory), format='cds')
    return table


def main():
    logging.basicConfig(level=logging.DEBUG)

    table = get_data()
    create_ch_table('vsx.vsx', table)
