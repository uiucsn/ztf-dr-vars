import logging
import os
from argparse import ArgumentParser
from dataclasses import dataclass
from functools import partial, reduce
from itertools import chain
from operator import and_
from typing import Callable, Optional

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True, eq=False)
class VarTypeColumn:
    name: str
    converter: Callable = str

    def __repr__(self):
        return self.name


@dataclass(frozen=True, eq=False)
class Catalog:
    @staticmethod
    def none_mask(*_args, **_kwargs):
        return slice(None)

    filename: str
    var_type_column: VarTypeColumn
    mask: Callable = none_mask


CATALOGS = {
    'sdss-vars': Catalog(
        filename='ztf-sdss-vars.csv.xz',
        var_type_column=VarTypeColumn('bhatti_vartype'),
    ),
    'sdss-candidates': Catalog(
        filename='ztf-sdss.csv.xz',
        var_type_column=VarTypeColumn('segue_tags'),
    ),
    'asassn': Catalog(
        filename='ztf-asassn.csv.xz',
        var_type_column=VarTypeColumn('asassn_type'),
    ),
    'vsx': Catalog(
        filename='ztf-vsx.csv.xz',
        var_type_column=VarTypeColumn(
            'type',
            # Take main type only
            converter=lambda s: s.split('/', maxsplit=1)[0],
        ),
        # : and | are uncertainty flags and + is for multiple types
        mask=lambda table: ~table['var_type'].str.contains(r'[:|+]', regex=True, na=False),
    ),
}


BAND_NAMES = {
    1: 'g',
    2: 'r',
    3: 'i',
}

COLORS = {
    1: 'green',
    2: 'red',
    3: 'black',
}

VAR_TYPE_THRESHOLDS = (
    {1: 50, 2: 50, 3: 0},
    {1: 100, 2: 100, 3: 0},
    {1: 50, 2: 50, 3: 50},
)

MAGN_BINS = np.arange(15, 23, 0.5)
OBS_COUNT_BINS = np.arange(0, 2000, 25)


def str_to_array(s, dtype=float):
    return np.fromstring(s[1:-1], sep=',', dtype=dtype)


def load_data(catalog, data_path):
    logging.info(f'Loading data for {catalog}')
    cat = CATALOGS[catalog]
    path = os.path.join(data_path, cat.filename)
    array_cols = ('mag', 'filter',)
    converters = dict.fromkeys(array_cols, str_to_array)
    converters['filter'] = partial(str_to_array, dtype=np.uint8)
    converters[cat.var_type_column.name] = cat.var_type_column.converter
    na_values = {cat.var_type_column.name: ()}
    cols = (cat.var_type_column.name,) + array_cols
    table = pd.read_csv(path, usecols=cols, converters=converters, na_values=na_values,)
    table.rename(columns={cat.var_type_column.name: 'var_type'}, inplace=True, errors='raise')
    table = table[cat.mask(table)]
    return table


def plot_mean_mag(table, catalog, fig_path, bands=tuple(BAND_NAMES)):
    logging.info(f'Plotting mean magnitude histogram for {catalog}')
    plt.title(catalog)
    plt.xlabel('Mean magnitude')
    plt.yscale('log')
    for band in bands:
        data = [np.mean(m[fltr == band]) for _, (fltr, m) in table[['filter', 'mag']].iterrows()]
        plt.hist(data, bins=MAGN_BINS, color=COLORS[band], histtype='step', label=BAND_NAMES[band])
    plt.ylim([1.0, None])
    plt.legend()
    path = os.path.join(fig_path, f'{catalog}_mean_mag.png')
    logging.info(f'Saving figure to {path}')
    plt.savefig(path)
    plt.close()


def obs_count_column_name(band):
    band_name = BAND_NAMES[band]
    return f'obs_count_{band_name}'


def plot_obs_count(table, catalog, fig_path, bands=tuple(BAND_NAMES)):
    logging.info(f'Plotting observation count histogram for {catalog}')
    plt.title(catalog)
    plt.xlabel('Observation count')
    plt.yscale('log')
    for band in bands:
        plt.hist(
            table[obs_count_column_name(band)],
            bins=OBS_COUNT_BINS,
            color=COLORS[band],
            histtype='step',
            label=BAND_NAMES[band],
        )
    plt.xlim([1.0, None])
    plt.ylim([1.0, None])
    plt.legend()
    path = os.path.join(fig_path, f'{catalog}_obs_count.png')
    logging.info(f'Saving figure to {path}')
    plt.savefig(path)
    plt.close()


def plot_var_types_chart(table, catalog, fig_path):
    logging.info(f'Plotting variable types chart for {catalog}')

    var_types, counts = np.unique(table['var_type'], return_counts=True)
    x = np.arange(var_types.shape[0])
    coords = dict(zip(var_types, x))
    logging.debug(f'{coords}')

    fig, ax = plt.subplots(figsize=(24, 12), dpi=300)

    ax.set_title(catalog)
    ax.set_ylabel('Count')
    ax.set_yscale('log')
    width = 0.8
    ax.bar(x, counts, width=width, label='all')
    for threshold in VAR_TYPE_THRESHOLDS:
        width *= 0.8
        idx = reduce(
            and_,
            (table[obs_count_column_name(band)] >= min_value for band, min_value in threshold.items()),
        )
        var_types_sub, counts_sub = np.unique(table['var_type'][idx], return_counts=True)
        x_sub = [coords[v] for v in var_types_sub]
        ax.bar(
            x_sub,
            counts_sub,
            width=width,
            label=', '.join(rf'$N_{BAND_NAMES[band]} \geq {v:d}$' for band, v in threshold.items()),
        )
    ax.set_ylim([1, None])
    ax.set_xticks(x)
    ax.set_xticklabels(var_types, rotation='vertical')
    ax.legend()

    path = os.path.join(fig_path, f'{catalog}_var-types.png')
    logging.info(f'Saving figure to {path}')
    fig.savefig(path)
    plt.close(fig)


def plot(catalog, data_path, fig_path):
    table = load_data(catalog, data_path=data_path)

    bands = sorted(set(chain.from_iterable(table['filter'])))
    for band in bands:
        table[obs_count_column_name(band)] = [np.count_nonzero(fltr == band) for fltr in table['filter']]

    plot_mean_mag(table, catalog, fig_path=fig_path, bands=bands)
    plot_obs_count(table, catalog, fig_path=fig_path, bands=bands)
    plot_var_types_chart(table, catalog, fig_path=fig_path)


def parse_args():
    parser = ArgumentParser('Plot statistics on cross-matched objects')
    parser.add_argument('-c', '--cat', default=tuple(CATALOGS), nargs='+', choices=CATALOGS,
                        help='catalogs to use')
    parser.add_argument('-d', '--data', default='https://static.rubin.science/',
                        help='data root, could be local path or HTTP URL (URL IS BROKEN FOR VSX DUE TO AN ISSUE WITH PANDAS)')
    parser.add_argument('-f', '--fig', default='.', help='folder to save figures')
    args = parser.parse_args()
    return args


def main():
    logging.basicConfig(level=logging.DEBUG)

    cli_args = parse_args()
    logging.debug(f'{cli_args = }')
    os.makedirs(cli_args.fig, exist_ok=True)

    for catalog in cli_args.cat:
        plot(catalog, data_path=cli_args.data, fig_path=cli_args.fig)
