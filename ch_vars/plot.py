import dataclasses
import logging
import os
import re
from argparse import ArgumentParser
from copy import copy
from functools import partial, reduce
from itertools import chain
from operator import and_
from typing import Callable, Optional, Union

import george
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gatspy.periodic import LombScargleMultibandFast

from ch_vars.vsx import VSX_TYPE_MAP


@dataclasses.dataclass(frozen=True)
class PeriodRange:
    min_period: float = 30. / 3600.
    max_period: float = np.inf

    @property
    def range(self):
        return self.min_period, self.max_period


@dataclasses.dataclass(frozen=True, eq=False)
class VarTypeColumn:
    name: str
    converter: Callable = str
    skip_value: Union[str, re.Pattern, None] = None
    map: Optional[Callable] = None

    def __repr__(self):
        return self.name


@dataclasses.dataclass(frozen=True, eq=False)
class Catalog:
    filename: str
    id_column: str
    var_type_column: VarTypeColumn
    periodic_types: dict = dataclasses.field(default_factory=dict)


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
        periodic_types={
            'M': PeriodRange(min_period=50),
            'RRAB': PeriodRange(min_period=0.05, max_period=5),
        }
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
        periodic_types={
            'Cepheid': PeriodRange(min_period=0.1, max_period=200),
            'Eclipsing': PeriodRange(min_period=0.01, max_period=2000),
            'Ellipsoidal': PeriodRange(min_period=0.01, max_period=1000),
            'Heatbeat': PeriodRange(min_period=0.5),
            'Mira': PeriodRange(min_period=50),
            'R': PeriodRange(max_period=10),
            'RS CVn': PeriodRange(min_period=10),
            'RR Lyr': PeriodRange(min_period=0.05, max_period=5),
            'ZZ Ceti': PeriodRange(max_period=0.1),
            'γ Dor': PeriodRange(min_period=0.1, max_period=10),
            'δ Sct': PeriodRange(max_period=5),
        },
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


BAND_WAVELENGTHS = {
    1: 4722.74,
    2: 6339.61,
    3: 7886.13,
}

VAR_TYPE_THRESHOLDS = (
    {1: 50, 2: 50, 3: 0},
    {1: 100, 2: 100, 3: 0},
    {1: 50, 2: 50, 3: 50},
)

MAGN_BINS = np.arange(15, 23, 0.5)
OBS_COUNT_BINS = np.arange(0, 2000, 25)


def fold_lc(obj, period_range):
    lsmf = LombScargleMultibandFast(fit_period=True)
    period_range = period_range[0], min(period_range[1], np.ptp(obj['mjd']))
    lsmf.optimizer.period_range = period_range
    lsmf.fit(obj['mjd'], obj['mag'], obj['magerr'], obj['filter'])
    period = lsmf.best_period
    phase = obj['mjd'] % period / period
    records = copy(obj.item()) + (phase, period,)
    dtype = obj.dtype.descr + [('phase', object), ('period', float)]
    folded = np.rec.fromrecords(records, dtype=dtype)
    return folded


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
            ls='', marker='x', color=COLORS[band],
            label=BAND_NAMES[band],
        )
    ax.legend(loc='upper right')


def approx_periodic(lc) -> Optional:
    _, idx = np.unique(lc['phase'], return_index=True)
    if idx.size < 10:
        return None
    phase = lc['phase'][idx]
    mag = lc['mag'][idx]
    magerr = lc['magerr'][idx]
    mean = np.average(mag, weights=magerr**-2)
    mag -= mean
    kernel = george.kernels.ExpSine2Kernel(gamma=1.0, log_period=0.0)
    # No fitting => no reason to freeze
    # kernel.freeze_parameter('k2:log_period')
    gp = george.GP(kernel)
    gp.compute(phase, magerr)

    def f(x):
        return gp.predict(mag, x, return_cov=False, return_var=False) + mean

    return f


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
                label=label, alpha=0.2,
            )
        spline = approx_periodic(lc)
        if spline is None:
            continue
        ph = np.linspace(0, 1.0, 1024)
        interp = spline(ph)
        for repeat in range(-1, 2):
            ax.plot(ph + repeat, interp, '-', color=COLORS[band], label='')
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


def plot_lc_examples(table, catalog, fig_path, bands=tuple(BAND_NAMES), n_per_type=9, rng=None):
    n_rows = n_columns = int(np.sqrt(n_per_type))
    if n_rows * n_columns != n_per_type:
        msg = f'n_per_type must be a square of a natural number, not {n_per_type}'
        logging.warning(msg)
        raise ValueError(msg)

    cat = CATALOGS[catalog]

    rng = np.random.default_rng(rng)

    var_types, inverse_idx = np.unique(table['var_type'], return_inverse=True)

    for type_idx, t in enumerate(var_types):
        positions = np.where(inverse_idx == type_idx)[0]
        try:
            object_positions = rng.choice(positions, n_per_type, replace=False)
        except ValueError:
            msg = f'n_per_type is {n_per_type} but there are only {positions.size} of objects of type {t}'
            logging.warning(msg)
            continue
        objects = table.iloc[object_positions].to_records().reshape(n_rows, n_columns)

        plot_lcs(
            objects,
            path=os.path.join(fig_path, f'{catalog}_{t}.png'),
            suptitle=f'{t} objects from {catalog}',
            bands=bands,
            ax_plot=plot_lc,
        )

        try:
            period_range = cat.periodic_types[t]
        except KeyError:
            pass
        else:
            folded_objects = np.array(
                [fold_lc(obj, period_range=period_range.range) for obj in objects.flat]
            ).reshape(objects.shape)
            plot_lcs(
                folded_objects,
                path=os.path.join(fig_path, f'{catalog}_{t}_folded.png'),
                suptitle=f'{t} objects from {catalog}',
                bands=bands,
                ax_plot=plot_folded,
            )


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
