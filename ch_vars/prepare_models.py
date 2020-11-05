import importlib.resources
import logging
import os
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
from ch_vars.common import str_to_array


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


class PrepareVsxFolded:
    vizier_id = 'OID'

    def __init__(self, data_dir, cache_dir):
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

    def __call__(self, type_ids, output_dir, n_approx=128):
        phase = np.linspace(0, 1, n_approx, endpoint=False)
        all_ids = frozenset.union(*type_ids.values())
        ztf_data = self.get_ztf_data(all_ids)
        for t, ids in type_ids.items():
            logging.info(f'Prepare {t}')
            bands = tuple(band for band, v in self.cat.types_plot_all[t].threshold.items() if v > 0)
            vsx_data = self.get_vsx_by_oids(ids)
            vsx_data.rename_column(self.vizier_id, self.cat.id_column)
            logging.debug(f'{vsx_data["RAJ2000"].data.data}')
            vsx_data['distance'] = self.get_distance(vsx_data['RAJ2000'].data.data, vsx_data['DEJ2000'].data.data)
            vsx_data['Egr'] = self.get_egr(
                vsx_data['RAJ2000'].data.data,
                vsx_data['DEJ2000'].data.data,
                vsx_data['distance'].data,
            )

            df = vsx_data.to_pandas(index=self.cat.id_column).join(ztf_data[ztf_data.index.isin(ids)])
            folded_objects = [self.fold_lc(obj, period_range=self.cat.periodic_types[t].range)
                              for obj in df.to_records()]
            df['period'] = [folded['period'] for folded in folded_objects]
            df['phase_model'] = [phase] * df.shape[0]
            models = {band: [] for band in bands}
            for folded, egr in zip(folded_objects, vsx_data['Egr']):
                for band in bands:
                    idx = folded['filter'].item() == band
                    folded['mag'] -= self.extinction_coeff[band] * egr

                    lc = {
                        'phase': folded['phase'].item()[idx],
                        'mag': folded['mag'].item()[idx],
                        'magerr': folded['magerr'].item()[idx]
                    }
                    m = approx_periodic(lc)(phase) - self.extinction_coeff[band] * egr
                    models[band].append(m)
            for band in bands:
                df[f'mag_folded_model_{BANDS[band]}'] = models[band]
            df.to_csv(os.path.join(output_dir, f'{t}.csv'))


def prepare_vsx_folded():
    import ch_vars.data.good_vsx_folded as data

    good_ids = get_ids(data)
    pvf = PrepareVsxFolded('/var/www/html', 'cache')
    pvf(good_ids, output_dir='output')


def main():
    logging.basicConfig(level=logging.DEBUG)
    prepare_vsx_folded()


if __name__ == '__main__':
    main()
