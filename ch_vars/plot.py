import logging
import os
import re
from argparse import ArgumentParser
from dataclasses import dataclass
from functools import partial, reduce
from itertools import chain
from operator import and_
from typing import Callable, Dict, Optional, Union

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ch_vars.vsx import VSX_TYPE_MAP


@dataclass(frozen=True, eq=False)
class VarTypeColumn:
    name: str
    converter: Callable = str
    skip_value: Union[str, re.Pattern, None] = None
    map: Optional[Callable] = None

    def __repr__(self):
        return self.name


@dataclass(frozen=True, eq=False)
class Catalog:
    filename: str
    id_column: str
    var_type_column: VarTypeColumn


CATALOGS = {
    'sdss-vars': Catalog(
        filename='ztf-sdss-vars.csv.xz',
        id_column='sdss_name',
        var_type_column=VarTypeColumn('bhatti_vartype'),
    ),
    'sdss-candidates': Catalog(
        filename='ztf-sdss.csv.xz',
        id_column='sdss_name',
        var_type_column=VarTypeColumn('segue_tags'),
    ),
    'asassn': Catalog(
        filename='ztf-asassn.csv.xz',
        id_column='asassn_name',
        var_type_column=VarTypeColumn('asassn_type'),
    ),
    'vsx': Catalog(
        filename='ztf-vsx.csv.xz',
        id_column='vsx_oid',
        var_type_column=VarTypeColumn(
            'type',
            converter=lambda s: s.split('/')[0],
            skip_value=re.compile(r'[:|+]'),
            map=lambda x: VSX_TYPE_MAP[x],
        ),
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
    array_cols = ('mjd', 'mag', 'magerr', 'filter',)
    converters = dict.fromkeys(array_cols, str_to_array)
    converters['filter'] = partial(str_to_array, dtype=np.uint8)
    converters[cat.var_type_column.name] = cat.var_type_column.converter
    na_values = {cat.var_type_column.name: ()}
    cols = (cat.id_column, cat.var_type_column.name,) + array_cols
    table = pd.read_csv(path, usecols=cols, converters=converters, na_values=na_values,)
    table.rename(columns={cat.id_column: 'id', cat.var_type_column.name: 'var_type'}, inplace=True, errors='raise')
    if cat.var_type_column.skip_value is not None:
        idx = table['var_type'].str.contains(
            cat.var_type_column.skip_value,
            regex=isinstance(cat.var_type_column.skip_value, re.Pattern)
        )
        table['var_type'][idx] = None
    table = table[~table['var_type'].isna()]
    if cat.var_type_column.map is not None:
        try:
            table['var_type'] = table['var_type'].map(cat.var_type_column.map)
        except KeyError as e:
            dif = set(table['var_type']) - set(VSX_TYPE_MAP)
            raise RuntimeError(f'{dif}') from e
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

    fig, ax = plt.subplots(figsize=(12, 6))

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
    ax.set_xticklabels(var_types, rotation=60, ha='right')
    ax.legend()

    path = os.path.join(fig_path, f'{catalog}_var-types.png')
    logging.info(f'Saving figure to {path}')
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_lc_examples(table, catalog, fig_path, bands=tuple(BAND_NAMES), n_per_type=9, rng=None):
    n_rows = n_columns = int(np.sqrt(n_per_type))
    if n_rows * n_columns != n_per_type:
        msg = f'n_per_type must be a square of a natural number, not {n_per_type}'
        logging.warning(msg)
        raise ValueError(msg)

    rng = np.random.default_rng(rng)

    var_types, inverse_idx = np.unique(table['var_type'], return_inverse=True)

    for type_idx, t in enumerate(var_types):
        positions = np.where(inverse_idx == type_idx)[0]
        try:
            object_positions = rng.choice(positions, n_per_type, replace=False)
        except ValueError:
            msg = f'n_per_object is {n_per_type} but there are only {positions.size} of objects of type {t}'
            logging.warning(msg)
            continue

        fig, ax_ = plt.subplots(n_rows, n_columns, figsize=(n_rows * 4, n_columns * 4))
        fig.suptitle(f'{t} type objects from {catalog}')
        for ax, obj_idx in np.nditer([ax_, object_positions.reshape(n_rows, n_columns)], flags=['refs_ok']):
            ax = ax.item()
            obj = table.iloc[obj_idx]
            lcs = {band: {'mjd': obj['mjd'][idx], 'mag': obj['mag'][idx], 'magerr': obj['magerr'][idx]}
                   for band in bands
                   if np.count_nonzero(idx := obj['filter'] == band) > 0}
            ax.set_title(obj['id'])
            ax.set_xlabel('MJD')
            ax.set_ylabel('mag')
            ax.invert_yaxis()
            for band, lc in lcs.items():
                ax.errorbar(
                    lc['mjd'], lc['mag'], lc['magerr'],
                    ls='', marker='x', color=COLORS[band],
                    label=BAND_NAMES[band],
                )
            ax.legend(loc='upper right')
        path = os.path.join(fig_path, f'{catalog}_{t}.png')
        logging.info(f'Saving figure to {path}')
        fig.tight_layout()
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

    examples_path = os.path.join(fig_path, 'lc_examples')
    os.makedirs(examples_path, exist_ok=True)
    threshold = VAR_TYPE_THRESHOLDS[1]
    idx_path_threshold = reduce(
        and_,
        (table[obs_count_column_name(band)] >= min_value for band, min_value in threshold.items()),
    )
    plot_lc_examples(table[idx_path_threshold], catalog, fig_path=examples_path, rng=0)


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
