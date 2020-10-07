import logging

import astropy.table
from astropy.io import ascii

from ch_vars.common import *


def get_data():
    table_vars = ascii.read(
        'https://wbhatti.org/stuff/stripe82/stripe82-periodic-variables-jun-2012.csv',
        format='csv',
    )
    table_candidates = ascii.read(
        'https://wbhatti.org/stuff/stripe82/stripe82-probable-variables-jun-2012.csv.gz',
        format='csv',
    )
    # Somehow not all vars are in candidates list
    table = astropy.table.join(table_candidates, table_vars, keys=('uid',))

    table.rename_column('ra', 'radeg')
    table.rename_column('dec', 'dedeg')
    return table


def main():
    logging.basicConfig(level=logging.DEBUG)

    table = get_data()
    create_ch_table('sdss.stripe82_vars', table)
