import logging
import os
import re
import sys
from argparse import ArgumentParser
from functools import partial

import numpy as np
import pandas as pd

from ch_vars.catalogs import CATALOGS
from ch_vars.common import str_to_array, numpy_print_options
from ch_vars.vsx import VSX_TYPE_MAP


def get_passband_lcs(df, fltr):
    def apply(series):
        idx = series['filter'] == fltr
        lc = series[['mjd', 'mag', 'magerr']].apply(lambda x: x[idx])
        return lc

    lcs = df.apply(apply, axis='columns')
    return lcs


def min_nights(lc, value):
    nights = np.unique(lc['mjd'].astype(np.uint32))
    return nights.size >= value


def min_observations(lc, value):
    return lc['mjd'].size >= value


def min_std(lc, value):
    if lc['mag'].size < 2:
        return False
    return lc['mag'].std(ddof=1) >= value


def filtered_chunk(chunk, var_type_column):
    zg = get_passband_lcs(chunk, 1)
    zr = get_passband_lcs(chunk, 2)
    # zi = get_passband_lcs(chunk, 3)
    idx = (
        zg.apply(min_nights, axis=1, args=(10,))
        & zr.apply(min_nights, axis=1, args=(10,))
        & zg.apply(min_observations, axis=1, args=(50,))
        & zr.apply(min_observations, axis=1, args=(50,))
        & zg.apply(min_std, axis=1, args=(0.1,))
        & zr.apply(min_std, axis=1, args=(0.1,))
    )
    if var_type_column.skip_value is not None:
        idx &= ~chunk[var_type_column.name].str.contains(
            var_type_column.skip_value,
            regex=isinstance(var_type_column.skip_value, re.Pattern)
        )
    return chunk[idx]


def get_data(cat, catalog_path):
    float_array_cols = ('mjd', 'mag', 'magerr',)
    converters = dict.fromkeys(float_array_cols, str_to_array)
    converters['filter'] = partial(str_to_array, dtype=np.uint8)
    converters[cat.var_type_column.name] = cat.var_type_column.converter
    na_values = {cat.var_type_column.name: ()}

    reader = pd.read_csv(
        catalog_path,
        index_col=cat.id_column,
        converters=converters,
        na_values=na_values,
        low_memory=True,
        iterator=True,
        chunksize=1 << 14,
    )
    df = pd.concat(filtered_chunk(chunk, cat.var_type_column) for chunk in reader)

    df[cat.var_type_column.name] = df[cat.var_type_column.name].map(cat.var_type_column.map)

    return df


def prepare_vsx_data(cli_args):
    cat = CATALOGS['vsx']
    catalog_path = os.path.join(cli_args.data, cat.filename)
    df = get_data(cat, catalog_path)
    path = os.path.join(cli_args.output, 'ztf-vsx-for-ml.csv.bz2')
    with numpy_print_options(threshold=sys.maxsize):
        df.to_csv(path)



def parse_args():
    parser = ArgumentParser('Join variable types and perform light-curve cuts')
    parser.add_argument('-d', '--data', default='https://static.rubin.science/',
                        help='data root, could be local path or HTTP URL (URL IS BROKEN FOR VSX DUE TO AN ISSUE WITH PANDAS)')
    parser.add_argument('--cache', default=None, help='directory to use as cache location')
    parser.add_argument('-o', '--output', default='.', help='directory to save data')
    args = parser.parse_args()
    return args


def main():
    logging.basicConfig(level=logging.DEBUG)
    cli_args = parse_args()
    prepare_vsx_data(cli_args)
