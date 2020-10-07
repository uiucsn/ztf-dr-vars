import logging
import os
from functools import partial
from itertools import chain
from urllib.parse import urljoin

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_URL = 'https://static.rubin.science'
FILE_NAMES = {
    'sdss-candidates': 'ztf-sdss.csv.xz',
    'sdss-vars': 'ztf-sdss-vars.csv.xz',
    'asassn': 'ztf-asassn.csv.xz',
    'vsx': 'ztf-vsx.csv.xz',
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

MAGN_BINS = np.arange(15, 23, 0.5)
OBS_COUNT_BINS = np.arange(0, 2000, 25)

FIG_DIR = 'figs'


def str_to_array(s, dtype=float):
    return np.fromstring(s[1:-1], sep=',', dtype=dtype)


def load_data(catalog):
    logging.info(f'Loading data for {catalog}')
    url = urljoin(BASE_URL, FILE_NAMES[catalog])
    cols = ('mag', 'filter',)
    converters = dict.fromkeys(cols, str_to_array)
    converters['filter'] = partial(str_to_array, dtype=np.uint8)
    table = pd.read_csv(url, usecols=cols, converters=converters)
    return table


def plot_mean_mag(table, catalog, bands=tuple(BAND_NAMES)):
    logging.info(f'Plotting mean magnitude histogram for {catalog}')
    plt.title(catalog)
    plt.xlabel('Mean magnitude')
    plt.yscale('log')
    for band in bands:
        data = [np.mean(m[fltr == band]) for _, (fltr, m) in table[['filter', 'mag']].iterrows()]
        plt.hist(data, bins=MAGN_BINS, color=COLORS[band], histtype='step', label=BAND_NAMES[band])
    plt.ylim([1.0, None])
    plt.legend()
    path = os.path.join(FIG_DIR, f'{catalog}_mean_mag.png')
    logging.info(f'Saving figure to {path}')
    plt.savefig(path)
    plt.close()


def plot_obs_count(table, catalog, bands=tuple(BAND_NAMES)):
    logging.info(f'Plotting observation count histogram for {catalog}')
    plt.title(catalog)
    plt.xlabel('Observation count')
    plt.yscale('log')
    for band in bands:
        data = [np.count_nonzero(fltr == band) for fltr in table['filter']]
        plt.hist(data, bins=OBS_COUNT_BINS, color=COLORS[band], histtype='step', label=BAND_NAMES[band])
    plt.xlim([1.0, None])
    plt.ylim([1.0, None])
    plt.legend()
    path = os.path.join(FIG_DIR, f'{catalog}_obs_count.png')
    logging.info(f'Saving figure to {path}')
    plt.savefig(path)
    plt.close()


def plot(catalog):
    table = load_data(catalog)
    bands = sorted(set(chain.from_iterable(table['filter'])))
    plot_mean_mag(table, catalog, bands)
    plot_obs_count(table, catalog, bands)


def main():
    logging.basicConfig(level=logging.DEBUG)

    os.makedirs(FIG_DIR, exist_ok=True)

    for catalog in FILE_NAMES:
        plot(catalog)
    # plot('sdss')
