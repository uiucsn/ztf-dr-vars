import importlib.resources
import logging
import os
import sys
from argparse import ArgumentParser
from datetime import datetime
from functools import partial

import astropy.table
import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astroquery.vizier import Vizier
from dustmaps import bayestar
from joblib import Memory

from ch_vars.approx import approx_periodic, fold_lc
from ch_vars.catalogs import CATALOGS
from ch_vars.common import str_to_array, numpy_print_options
from ch_vars.vsx import VSX_JOINED_TYPES


NOW = datetime.now()


EXTRAGALACTIC_DISTANCE = 1e6 * u.pc


BANDS = {1: 'g', 2: 'r', 3: 'i'}


def get_ids(package):
    def read_file(fname):
        with importlib.resources.open_text(package, fname) as fh:
            return frozenset(map(int, fh.read().splitlines()))

    filenames = (fname for fname in importlib.resources.contents(package) if fname.endswith('.dat'))
    ids = {os.path.splitext(name)[0]: read_file(name) for name in sorted(filenames)}
    return ids


class _DustMap:
    bayestar_version = 'bayestar2019'
    bayestar_r = {1: 3.518, 2: 2.617, 3: 1.971}

    def __init__(self, cache_dir):
        bayestar.config['data_dir'] = cache_dir
        bayestar.fetch(version=self.bayestar_version)
        self.bayestar = bayestar.BayestarQuery(version=self.bayestar_version)


class DustMap(_DustMap):
    _objects = {}

    def __new__(cls, cache_dir):
        if cache_dir not in cls._objects:
            cls._objects[cache_dir] = _DustMap(cache_dir)
        return cls._objects[cache_dir]


def beyestar_get(ra, dec, distance, cache_dir):
    coords = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, distance=distance*u.pc)
    return DustMap(cache_dir).bayestar(coords)


def get_ztf_data(ids, catalog_path, id_column):
    array_cols = ('mjd', 'mag', 'magerr', 'filter',)
    converters = dict.fromkeys(array_cols, str_to_array)
    converters['filter'] = partial(str_to_array, dtype=np.uint8)
    reader = pd.read_csv(
        catalog_path,
        usecols=(id_column,) + array_cols,
        converters=converters,
        index_col=id_column,
        low_memory=True,
        iterator=True,
        chunksize=1 << 14,
    )
    df = pd.concat(chunk[chunk.index.isin(ids)] for chunk in reader)
    assert len(ids) == df.shape[0]
    return df


def get_vsx_by_oids(vsx_oids):
    vizier = Vizier()
    rows = [vizier.query_constraints(catalog='B/vsx/vsx', OID=oid)[0][0] for oid in vsx_oids]
    table = astropy.table.vstack(rows)
    return table


def get_distance(ra, dec, search_radius=1*u.arcsec):
    vizier = Vizier()
    coords = SkyCoord(ra=ra, dec=dec, unit=u.deg)
    response = vizier.query_region(coords, radius=search_radius, catalog='I/347/gaia2dis')
    distance = u.Quantity([EXTRAGALACTIC_DISTANCE] * coords.size)
    if not response:
        return distance
    table = response[0]
    coord_idx, table_idx = np.unique(table['_q'] - 1, return_inverse=True)
    for i, coord_i in enumerate(coord_idx):
        distance[coord_i] = table[table_idx == i][0]['rest'] * table['rest'].unit
    return distance


class VsxDataGetter:
    vizier_id = 'OID'

    def __init__(self, data_dir, cache_dir, type_ids):
        self.cat = CATALOGS['vsx']
        self.memory = Memory(location=cache_dir)
        dustmaps_cache_dir = os.path.join(cache_dir, 'dustmaps')
        self.get_egr = partial(self.memory.cache(beyestar_get), cache_dir=dustmaps_cache_dir)
        self.extinction_coeff = DustMap.bayestar_r
        self.vizier = Vizier()
        catalog_path = os.path.join(data_dir, self.cat.filename)
        self.get_ztf_data = partial(
            self.memory.cache(get_ztf_data),
            catalog_path=catalog_path,
            id_column=self.cat.id_column
        )
        self.fold_lc = self.memory.cache(fold_lc)
        self.get_vsx_by_oids = self.memory.cache(get_vsx_by_oids)
        self.get_distance = self.memory.cache(get_distance)

        self.type_ids = type_ids.copy()
        self.ztf_data = self.get_ztf_data(self.all_ids)

    @property
    def all_ids(self):
        return frozenset.union(*self.type_ids.values())

    def update_ztf_data(self, var_type, ids):
        if self.type_ids.get(var_type, None) == ids:
            return
        self.type_ids[var_type] = ids
        self.ztf_data = self.get_ztf_data(self.all_ids)


class VsxFoldedModel:
    max_abs_b = 10

    def __init__(self, data, var_type, ids):
        self.data = data
        self.var_type = var_type
        self.ids = ids
        self.id_column = self.data.cat.id_column
        self.data.update_ztf_data(var_type, ids)
        self.bands = tuple(band for band, v in self.data.cat.types_plot_all[var_type].threshold.items() if v > 0)
        self.vsx_data = self.data.get_vsx_by_oids(ids)
        self.vsx_data.rename_column(self.data.vizier_id, self.id_column)
        self.vsx_data['distance'] = self.data.get_distance(
            self.vsx_data['RAJ2000'].data.data,
            self.vsx_data['DEJ2000'].data.data,
        )
        self.vsx_data['Egr'] = self.data.get_egr(
            self.vsx_data['RAJ2000'].data.data,
            self.vsx_data['DEJ2000'].data.data,
            self.vsx_data['distance'].data,
        )
        galactic = SkyCoord(ra=self.vsx_data['RAJ2000'], dec=self.vsx_data['DEJ2000']).galactic
        self.vsx_data['l'] = galactic.l
        self.vsx_data['b'] = galactic.b

        self.df = self.vsx_data.to_pandas(index=self.id_column).join(self.data.ztf_data[self.data.ztf_data.index.isin(ids)])
        self.folded_objects = [self.data.fold_lc(obj, period_range=self.data.cat.periodic_types[var_type].range)
                               for obj in self.df.to_records()]
        self.df['period'] = [folded['period'] for folded in self.folded_objects]
        self.approx_funcs = {i: {} for i in self.df.index}
        self.extinction = {i: {} for i in self.df.index}
        for i, folded, egr in zip(self.df.index, self.folded_objects, self.vsx_data['Egr']):
            for band in self.bands:
                idx = folded['filter'].item() == band
                lc = {
                    'phase': folded['phase'].item()[idx],
                    'mag': folded['mag'].item()[idx],
                    'magerr': folded['magerr'].item()[idx]
                }
                self.approx_funcs[i][band] = approx_periodic(lc)
                self.extinction[i][band] = self.data.extinction_coeff[band] * egr

    def model_column(self, band):
        return f'mag_folded_model_{BANDS[band]}'

    def with_approxed(self, n_approx, endpoint=False):
        phase = np.linspace(0.0, 1.0, n_approx, endpoint=endpoint)
        df = self.df.copy(deep=True)
        df['phase_model'] = [phase] * df.shape[0]
        df['folded_time_model'] = [phase * period for period in df['period']]
        for band in self.bands:
            df[self.model_column(band)] = pd.Series({i: self.approx_funcs[i][band](phase) - self.extinction[i][band]
                                                     for i in df.index})
        return df

    def to_csv(self, path, n_approx=128):
        if os.path.isdir(path):
            path = os.path.join(path, f'{self.var_type}.csv.bz2')
        with numpy_print_options(threshold=sys.maxsize):
            self.with_approxed(n_approx).to_csv(path)

    def to_lclib(self, path, n_approx=128):
        if os.path.isdir(path):
            path = os.path.join(path, f'{self.var_type}.lclib')

        df = self.with_approxed(n_approx, endpoint=True)

        with open(path, 'w') as fh:
            fh.write(
                'SURVEY: ZTF\n'
                f'FILTERS: {"".join(BANDS[band] for band in self.bands)}\n'
                f'MODEL: ZTFDR3VSX{self.var_type.upper()}\n'  # CHECKME
                'MODEL_PARNAMES: VSXOID,PERIOD\n'  # CHECKME
                'RECUR_CLASS: RECUR-PERIODIC\n'
                f'NEVENT: {df.shape[0]}\n'
                '\n'
            )
            fh.write(
                f'COMMENT: Created on {NOW.date()} by Konstantin Malanchev\n'
                'COMMENT: Based on ZTF DR3 light curves cross-matched with VSX 2020.10 edition\n'
                f'COMMENT: VSX types used: {", ".join(sorted(VSX_JOINED_TYPES[self.var_type].types))}\n'
                'COMMENT: Periodic Gaussian process is used to approximate folded light curves\n'
                'COMMENT: Bayestar 2019 was used for deredding, Green, Schlafly, Finkbeiner et al. (2019)\n'
                'COMMENT: VSXOID is the VSX object identifier\n'
                'COMMENT: PERIOD is the used period, in days\n'
                '\n'
            )
            for i_event, (vsx_oid, row) in enumerate(df.iterrows()):
                fh.write(
                    f'START_EVENT: {i_event}\n'
                    f'NROW: {n_approx} RA: {row["RAJ2000"]} DEC: {row["DEJ2000"]}\n'
                    f'PARVAL: {vsx_oid},{row.period:.6g}\n'
                    f'ANGLEMATCH_b: {self.max_abs_b}\n'  # CHECKME
                )
                for i in range(n_approx):
                    time = row['folded_time_model'][i]
                    fh.write(f'S: {time:7.4f}')
                    for band in self.bands:
                        fh.write(f' {row[self.model_column(band)][i]:.3f}')
                    fh.write('\n')
                fh.write(
                    f'END_EVENT: {i_event}\n'
                    '\n'
                )


def prepare_vsx_folded(cli_args):
    import ch_vars.data.good_vsx_folded as data

    good_ids = get_ids(data)
    data_getter = VsxDataGetter(cli_args.data, cli_args.cache, good_ids)
    for var_type, ids in good_ids.items():
        model = VsxFoldedModel(data_getter, var_type, ids)
        if cli_args.csv:
            model.to_csv(cli_args.output)
        if cli_args.lclib:
            model.to_lclib(cli_args.output)


def parse_args():
    parser = ArgumentParser('Create Gaussian Process approximated models')
    parser.add_argument('--csv', action='store_true', help='save into .csv.bz2 files')
    parser.add_argument('--lclib', action='store_true', help='save in SNANA LCLIB format')
    parser.add_argument('-d', '--data', default='https://static.rubin.science/',
                        help='data root, could be local path or HTTP URL (URL IS BROKEN FOR VSX DUE TO AN ISSUE WITH PANDAS)')
    parser.add_argument('--cache', default=None, help='directory to use as cache location')
    parser.add_argument('-o', '--output', default='.', help='directory to save models')
    args = parser.parse_args()
    return args


def main():
    logging.basicConfig(level=logging.DEBUG)
    cli_args = parse_args()
    prepare_vsx_folded(cli_args)


if __name__ == '__main__':
    main()
