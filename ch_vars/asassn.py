import importlib.resources

from astropy.io import ascii

from ch_vars.common import *
from ch_vars import data


ASASSN_CATALOG_FILENAME = 'asassn-catalog-20191020.csv.bz2'


def get_data():
    logging.info('Loading ASAS-SN variable catalog from the file')
    with importlib.resources.open_binary(data, ASASSN_CATALOG_FILENAME) as fh:
        table = ascii.read(fh, format='csv')
    table.rename_column('raj2000', 'radeg')
    table.rename_column('dej2000', 'dedeg')
    return table


def main():
    logging.basicConfig(level=logging.DEBUG)

    table = get_data()
    create_ch_table('asassn_var.asassn_var_meta', table)
