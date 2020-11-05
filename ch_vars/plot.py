import logging
import os
import re
from argparse import ArgumentParser
from functools import partial, reduce
from itertools import chain
from operator import and_

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Memory

from ch_vars.approx import *
from ch_vars.catalogs import CATALOGS
from ch_vars.common import str_to_array
from ch_vars.vsx import VSX_TYPE_MAP


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


BAND_WAVELENGTHS = {
    1: 4722.74,
    2: 6339.61,
    3: 7886.13,
}

VAR_TYPE_THRESHOLDS = (
    {1: 100, 2: 100, 3: 0},
    {1: 50, 2: 50, 3: 25},
    {1: 50, 2: 50, 3: 50},
)

MAGN_BINS = np.arange(15, 23, 0.5)
OBS_COUNT_BINS = np.arange(0, 2000, 25)


def obs_count_column_name(band):
    band_name = BAND_NAMES[band]
    return f'obs_count_{band_name}'


def plot_lc(ax, obj, *, bands):
    lcs = {band: {'mjd': obj['mjd'].item()[idx], 'mag': obj['mag'].item()[idx], 'magerr': obj['magerr'].item()[idx]}
           for band in bands
           if np.count_nonzero(idx := obj['filter'].item() == band) > 0}
    ax.set_title(obj['id'])
    ax.set_xlabel('MJD')
    ax.set_ylabel('mag')
    ax.invert_yaxis()
    for band, lc in lcs.items():
        ax.errorbar(
            lc['mjd'], lc['mag'], lc['magerr'],
            ls='', marker='x', color=COLORS[band], alpha=0.3,
            label=BAND_NAMES[band],
        )
        spline = approx(lc)
        if spline is None:
            continue
        t = np.linspace(lc['mjd'].min(), lc['mjd'].max(), 128)
        interp = spline(t)
        ax.plot(t, interp, '-', color=COLORS[band], label='')
    ax.set_ylim(obj['mag'].item().max() + 2.0 * obj['magerr'].item().max(), obj['mag'].item().min() - 2.0 * obj['magerr'].item().max())
    ax.legend(loc='upper right')


def plot_folded(ax, obj, *, bands):
    lcs = {band: {'phase': obj['phase'].item()[idx], 'mag': obj['mag'].item()[idx], 'magerr': obj['magerr'].item()[idx]}
           for band in bands
           if np.count_nonzero(idx := obj['filter'].item() == band) > 0}
    ax.set_title(f'{obj["id"]}, P = {obj["period"]:.3g}')
    ax.set_xlabel('phase')
    ax.set_ylabel('mag')
    ax.invert_yaxis()
    for band, lc in lcs.items():
        for repeat in range(-1, 2):
            label = BAND_NAMES[band] if repeat == 0 else ''
            ax.errorbar(
                lc['phase'] + repeat, lc['mag'], lc['magerr'],
                ls='', marker='x', color=COLORS[band],
                label=label, alpha=0.1,
            )
        spline = approx_periodic(lc)
        if spline is None:
            continue
        ph = np.linspace(0, 1.0, 128)
        interp = spline(ph)
        for repeat in range(-1, 2):
            ax.plot(ph + repeat, interp, '-', color=COLORS[band], label='')
    ax.set_ylim(obj['mag'].item().max() + 2.0 * obj['magerr'].item().max(), obj['mag'].item().min() - 2.0 * obj['magerr'].item().max())
    ax.set_xlim([-0.2, 1.8])
    secax = ax.secondary_xaxis('top', functions=(lambda x: x * obj['period'], lambda x: x / obj['period']))
    secax.set_xlabel('Folded time, days')
    ax.legend(loc='upper right')


def plot_lcs(objs, *, path, suptitle, bands, ax_plot=plot_lc):
    n_rows, n_columns = objs.shape
    fig, ax_ = plt.subplots(n_rows, n_columns, figsize=(n_rows * 4, n_columns * 4))
    fig.suptitle(suptitle)
    for ax, obj in np.nditer([ax_, objs], flags=['refs_ok']):
        ax = ax.item()
        ax_plot(ax, obj, bands=bands)
    logging.info(f'Saving figure to {path}')
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_all_lc(objects, fig_path, bands=tuple(BAND_NAMES), ax_plot=plot_lc):
    for obj in objects:
        plot_lcs(
            np.array([[obj]]),
            path=os.path.join(fig_path, f'{obj["id"]}.png'),
            suptitle='',
            bands=bands,
            ax_plot=ax_plot,
        )


def threshold_idx(table, threshold):
    return reduce(
        and_,
        (table[obs_count_column_name(band)] >= min_value for band, min_value in threshold.items()),
    )


class Plot:
    def __init__(self, catalog, fig_path, cache_path=None, clear_cache=False):
        self.catalog_name = catalog
        self.cat = CATALOGS[self.catalog_name]
        self.fig_path = fig_path
        os.makedirs(self.fig_path, exist_ok=True)
        self.memory = Memory(location=cache_path)
        if clear_cache:
            self.memory.clear()
        self.fold_lc = self.memory.cache(fold_lc)

    def load_data(self, data_path):
        logging.info(f'Loading data for {self.catalog_name}')
        path = os.path.join(data_path, self.cat.filename)
        array_cols = ('mjd', 'mag', 'magerr', 'filter',)
        converters = dict.fromkeys(array_cols, str_to_array)
        converters['filter'] = partial(str_to_array, dtype=np.uint8)
        converters[self.cat.var_type_column.name] = self.cat.var_type_column.converter
        na_values = {self.cat.var_type_column.name: ()}
        cols = (self.cat.id_column, self.cat.var_type_column.name,) + array_cols
        table = pd.read_csv(path, usecols=cols, converters=converters, na_values=na_values, )
        table.rename(
            columns={self.cat.id_column: 'id', self.cat.var_type_column.name: 'var_type'},
            inplace=True,
            errors='raise',
        )
        if self.cat.var_type_column.skip_value is not None:
            idx = table['var_type'].str.contains(
                self.cat.var_type_column.skip_value,
                regex=isinstance(self.cat.var_type_column.skip_value, re.Pattern)
            )
            table['var_type'][idx] = None
        table = table[~table['var_type'].isna()]
        if self.cat.var_type_column.map is not None:
            try:
                table['var_type'] = table['var_type'].map(self.cat.var_type_column.map)
            except KeyError as e:
                dif = set(table['var_type']) - set(VSX_TYPE_MAP)
                raise RuntimeError(f'{dif}') from e
        return table

    def plot_mean_mag(self, table, bands=tuple(BAND_NAMES)):
        logging.info(f'Plotting mean magnitude histogram for {self.catalog_name}')
        plt.title(self.catalog_name)
        plt.xlabel('Mean magnitude')
        plt.yscale('log')
        for band in bands:
            data = [np.mean(m[fltr == band]) for _, (fltr, m) in table[['filter', 'mag']].iterrows()]
            plt.hist(data, bins=MAGN_BINS, color=COLORS[band], histtype='step', label=BAND_NAMES[band])
        plt.ylim([1.0, None])
        plt.legend()
        path = os.path.join(self.fig_path, f'{self.catalog_name}_mean_mag.png')
        logging.info(f'Saving figure to {path}')
        plt.savefig(path)
        plt.close()

    def plot_obs_count(self, table, bands=tuple(BAND_NAMES)):
        logging.info(f'Plotting observation count histogram for {self.catalog_name}')
        plt.title(self.catalog_name)
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
        path = os.path.join(self.fig_path, f'{self.catalog_name}_obs_count.png')
        logging.info(f'Saving figure to {path}')
        plt.savefig(path)
        plt.close()

    def plot_var_types_chart(self, table):
        logging.info(f'Plotting variable types chart for {self.catalog_name}')

        var_types, counts = np.unique(table['var_type'], return_counts=True)
        x = np.arange(var_types.shape[0])
        coords = dict(zip(var_types, x))
        logging.debug(f'{coords}')

        fig, ax = plt.subplots(figsize=(12, 6))

        ax.set_title(self.catalog_name)
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

        path = os.path.join(self.fig_path, f'{self.catalog_name}_var-types.png')
        logging.info(f'Saving figure to {path}')
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)

    def plot_lc_examples(self, table, threshold, bands=tuple(BAND_NAMES), n_per_type=9, rng=0):
        examples_path = os.path.join(self.fig_path, 'lc_examples')
        os.makedirs(examples_path, exist_ok=True)

        table = table[threshold_idx(table, threshold)]

        n_rows = n_columns = int(np.sqrt(n_per_type))
        if n_rows * n_columns != n_per_type:
            msg = f'n_per_type must be a square of a natural number, not {n_per_type}'
            logging.warning(msg)
            raise ValueError(msg)

        rng = np.random.default_rng(rng)

        var_types, inverse_idx = np.unique(table['var_type'], return_inverse=True)

        for type_idx, t in enumerate(var_types):
            positions = np.where(inverse_idx == type_idx)[0]
            if positions.size < n_per_type:
                msg = f'n_per_type is {n_per_type} but there are only {positions.size} of objects of type {t}'
                logging.warning(msg)
                continue
            object_positions = rng.choice(positions, n_per_type, replace=False)
            objects = table.iloc[object_positions].to_records().reshape(n_rows, n_columns)

            plot_lcs(
                objects,
                path=os.path.join(examples_path, f'{self.catalog_name}_{t}.png'),
                suptitle=f'{t} objects from {self.catalog_name}',
                bands=bands,
                ax_plot=plot_lc,
            )

            try:
                period_range = self.cat.periodic_types[t]
            except KeyError:
                pass
            else:
                folded_objects = np.array(
                    [self.fold_lc(obj, period_range=period_range.range) for obj in objects.flat]
                ).reshape(objects.shape)
                plot_lcs(
                    folded_objects,
                    path=os.path.join(examples_path, f'{self.catalog_name}_{t}_folded.png'),
                    suptitle=f'{t} objects from {self.catalog_name}',
                    bands=bands,
                    ax_plot=plot_folded,
                )

    def plot_types_all_lc(self, table):
        for t, tpa in self.cat.types_plot_all.items():
            idx = reduce(
                and_,
                (table[obs_count_column_name(band)] >= min_value for band, min_value in tpa.threshold.items()),
            )
            bands = sorted(band for band, v in tpa.threshold.items() if v > 0)
            objects = table[idx & (table['var_type'] == t)].to_records()
            if tpa.as_is:
                raise NotImplementedError
            if tpa.folded:
                folded_objects = np.array(
                    [self.fold_lc(obj, period_range=self.cat.periodic_types[t].range) for obj in objects]
                )
                lc_path = os.path.join(self.fig_path, f'{self.catalog_name}_{t}_folded')
                os.makedirs(lc_path, exist_ok=True)
                plot_all_lc(folded_objects, lc_path, bands=bands, ax_plot=plot_folded)

    def __call__(self, data_path):
        table = self.load_data(data_path=data_path)

        bands = sorted(set(chain.from_iterable(table['filter'])))
        for band in bands:
            table[obs_count_column_name(band)] = [np.count_nonzero(fltr == band) for fltr in table['filter']]

        self.plot_mean_mag(table, bands=bands)
        self.plot_obs_count(table, bands=bands)
        self.plot_var_types_chart(table)
        self.plot_lc_examples(table, threshold=VAR_TYPE_THRESHOLDS[0], bands=bands)
        self.plot_types_all_lc(table)


def parse_args():
    parser = ArgumentParser('Plot statistics on cross-matched objects')
    parser.add_argument('-c', '--cat', default=tuple(CATALOGS), nargs='+', choices=CATALOGS,
                        help='catalogs to use')
    parser.add_argument('-d', '--data', default='https://static.rubin.science/',
                        help='data root, could be local path or HTTP URL (URL IS BROKEN FOR VSX DUE TO AN ISSUE WITH PANDAS)')
    parser.add_argument('-f', '--fig', default='.', help='folder to save figures')
    parser.add_argument('--cache', default=None, help='directory to use as cache location')
    parser.add_argument('--clear-cache', default=False, action='store_true', help='clear cache')
    args = parser.parse_args()
    return args


def main():
    logging.basicConfig(level=logging.DEBUG)

    cli_args = parse_args()
    logging.debug(f'{cli_args = }')
    os.makedirs(cli_args.fig, exist_ok=True)

    for catalog in cli_args.cat:
        plot = Plot(catalog, fig_path=cli_args.fig, cache_path=cli_args.cache, clear_cache=cli_args.clear_cache)
        plot(data_path=cli_args.data)
