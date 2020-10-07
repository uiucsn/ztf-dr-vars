import logging

from astropy.io import ascii

from ch_vars.common import *


def get_data():
    table = ascii.read(
        'https://wbhatti.org/stuff/stripe82/stripe82-probable-variables-jun-2012.csv.gz',
        format='csv',
    )
    table.rename_column('ra', 'radeg')
    table.rename_column('dec', 'dedeg')
    return table


def main():
    logging.basicConfig(level=logging.DEBUG)

    table = get_data()
    create_ch_table('sdss.stripe82_candidates', table)
