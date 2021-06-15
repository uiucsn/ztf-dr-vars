import importlib.resources
import logging
import os
import re
import sys
from argparse import ArgumentParser
from datetime import datetime
from functools import partial

import astropy.constants as const
import astropy.io.ascii
import astropy.table
import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import Distance, SkyCoord
from astroquery.vizier import Vizier
from joblib import Memory
from pyvo.dal import TAPService
from scipy.integrate import simps
from scipy.optimize import least_squares
from scipy import stats

from ch_vars.approx import approx_periodic, fold_lc
from ch_vars.catalogs import CATALOGS
from ch_vars.common import BAND_NAMES, COLORS, greek_to_latin, str_to_array, numpy_print_options, LSST_BANDS,\
    LSST_COLORS
from ch_vars.data import lsst_band_transmission, ztf_band_transmission
from ch_vars.extinction import BayestarDustMap, bayestar_get, get_sfd_thin_disk_ebv, LSST_A_TO_EBV
from ch_vars.spatial_distr import MilkyWayDensityJuric2008
from ch_vars.vsx import VSX_JOINED_TYPES


NOW = datetime.now()


EXTRAGALACTIC_DISTANCE = 1e6 * u.pc


def get_ids(package):
    def read_file(fname):
        with importlib.resources.open_text(package, fname) as fh:
            return frozenset(map(int, fh.read().splitlines()))

    filenames = (fname for fname in importlib.resources.contents(package) if fname.endswith('.dat'))
    ids = {os.path.splitext(name)[0]: read_file(name) for name in sorted(filenames)}
    return ids


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


def get_distance(ra, dec, search_radius_arcsec=1):
    tap = TAPService('https://dc.zah.uni-heidelberg.de/tap')
    distance = []
    for ra_deg, dec_deg in zip(ra, dec):
        # source_id, ra, dec, r_med_geo, r_lo_geo, r_hi_geo, r_med_photogeo, r_lo_photogeo, r_hi_photogeo
        response = tap.search(f'''
            SELECT distance(ra, dec, {ra_deg}, {dec_deg}) as d, r_med_geo, r_med_photogeo
                FROM gedr3dist.main
                JOIN gaia.edr3lite USING (source_id)
                WHERE distance(ra, dec, {ra_deg}, {dec_deg}) < {search_radius_arcsec / 3600.0}
        ''')
        if not response:
            distance.append(EXTRAGALACTIC_DISTANCE)
            continue
        row = response.getrecord(np.argmin(response['d']))
        distance.append(row['r_med_photogeo'].item() * u.pc)
    return u.Quantity(distance)


HC_K = const.h.cgs.value * const.c.cgs.value / const.k_B.cgs.value


def bb_mag(transmission, m0, temperature):
    lmbd, r, norm = transmission
    flux = simps(
        x=lmbd,
        y=r / (lmbd**5 * np.expm1(HC_K / (lmbd * temperature))),
    )
    return m0 - 2.5 * np.log10(flux / norm)


def fit_temperature(curves):
    some_band, some_curve = next(iter(curves.items()))
    x = []
    temp_init = 6000.0
    m0_init = some_curve[0] - bb_mag(ztf_band_transmission(some_band), 0.0, temp_init)
    x0 = [m0_init, temp_init]
    for i in range(some_curve.size):
        result = least_squares(
            lambda x: [bb_mag(ztf_band_transmission(band), x[0], x[1]) - curve[i]
                       for band, curve in curves.items()],
            x0=x0,
            bounds=([-np.inf, 3000], [np.inf, 10000]),
        )
        assert result.success
        x.append(result.x)
    m0, t = np.array(x).T
    return m0, t


class VsxDataGetter:
    vizier_id = 'OID'

    def __init__(self, data_dir, cache_dir, type_ids):
        self.cat = CATALOGS['vsx']
        self.cache_dir = cache_dir
        self.memory = Memory(location=self.cache_dir)
        if self.cache_dir is None:
            self.dustmaps_cache_dir = None
        else:
            self.dustmaps_cache_dir = os.path.join(self.cache_dir, 'dustmaps')
        self.get_egr = partial(self.memory.cache(bayestar_get), cache_dir=self.dustmaps_cache_dir)
        self.extinction_coeff = BayestarDustMap.bayestar_r
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
        self.fit_temperature = self.memory.cache(fit_temperature)

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

    def __init__(self, data: VsxDataGetter, var_type, ids):
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
        self.vsx_data = self.vsx_data[self.vsx_data['distance'] != EXTRAGALACTIC_DISTANCE]
        self.vsx_data['distmod'] = Distance(self.vsx_data['distance'], unit=u.pc).distmod
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
        for i, folded, egr, distmod in zip(self.df.index, self.folded_objects, self.vsx_data['Egr'], self.vsx_data['distmod']):
            # if var_type == 'Cepheid':
            #     mean_m_g = np.mean(folded['mag'].item()[folded['filter'].item() == 1])
            #     # https://arxiv.org/pdf/0707.3144.pdf
            #     Mv = -3.932 - 2.819 * (np.log10(folded['period']) - 1.0)
            #     mv = Mv + distmod
            #     Av = mean_m_g - mv
            #     if Av > 0.0:
            #         eBV = Av / 3.1
            #         egr = eBV / 0.981
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
        return f'mag_folded_model_{BAND_NAMES[band]}'

    def lsst_model_column(self, band):
        return f'mag_folded_model_lsst_{band}'

    def with_approxed(self, n_approx, endpoint=False, max_egr=np.inf):
        phase = np.linspace(0.0, 1.0, n_approx, endpoint=endpoint)
        idx = self.df['Egr'] < max_egr
        df = self.df[idx].copy(deep=True)
        df['phase_model'] = [phase] * df.shape[0]
        df['folded_time_model'] = [phase * period for period in df['period']]
        for band in self.bands:
            df[self.model_column(band)] = pd.Series(
                {i: self.approx_funcs[i][band](phase) - self.extinction[i][band] - df.loc[i].distmod
                 for i in df.index}
            )
        m0_temp = np.array([
            self.data.fit_temperature({band: df.loc[i][self.model_column(band)] for band in self.bands})
            for i in df.index
        ])
        df['temperature'] = pd.Series({i: x[1] for i, x in zip(df.index, m0_temp)})
        for lsst_band in LSST_BANDS:
            df[self.lsst_model_column(lsst_band)] = pd.Series(
                {i: np.array([bb_mag(lsst_band_transmission(lsst_band), m0, temp) for m0, temp in x.T])
                 for i, x in zip(df.index, m0_temp)}
            )
        return df

    def plot_mean_hist(self, df, path):
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        fig.suptitle(self.var_type)
        ax.set_xlabel(r'Mean magnitude')
        for band in self.bands:
            mean = df[self.model_column(band)].apply(np.mean)
            ax.hist(mean, histtype='step', label=BAND_NAMES[band], color=COLORS[band])
        ax.legend()
        fig.savefig(path)
        plt.close(fig)

    def plot_lcs(self, df, path):
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        os.makedirs(path, exist_ok=True)

        for i, row in df.iterrows():
            fig, ax = plt.subplots()
            fig.suptitle(f'{self.var_type} : {i}')
            ax.set_xlabel(r'phase')
            ax.set_xlim([0, 1])
            ax.set_ylabel(r'magnitude')
            ax.invert_yaxis()

            idx_data = np.isin(row['filter'], self.bands)
            phase_data = row['mjd'][idx_data] % row['period'] / row['period']
            mag_data = (
                row['mag'][idx_data]
                - np.array([self.extinction[i][band] for band in row['filter'][idx_data]])
                - row['distmod']
            )
            color_data = np.vectorize(COLORS.get)(row['filter'][idx_data])
            ax.scatter(phase_data, mag_data, color=color_data, alpha=0.2, s=4)

            for band in self.bands:
                ax.plot(
                    row['phase_model'],
                    row[self.model_column(band)],
                    '--',
                    lw=1,
                    color=COLORS[band],
                    label=f'ZTF {BAND_NAMES[band]}'
                )

            for lsst_band in LSST_BANDS:
                ax.plot(
                    row['phase_model'],
                    row[self.lsst_model_column(lsst_band)],
                    '-',
                    lw=2,
                    color=LSST_COLORS[lsst_band],
                    label=f'LSST {lsst_band}'
                )

            ax.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
            fig.tight_layout()
            plt.savefig(os.path.join(path, f'{i}.png'), dpi=300)
            plt.close(fig)

    def plots(self, path, n_approx=128):
        os.makedirs(path, exist_ok=True)
        df = self.with_approxed(n_approx, endpoint=True)
        self.plot_mean_hist(df, os.path.join(path, f'abs_mag_hist_{self.var_type}.png'))
        self.plot_lcs(df, os.path.join(path, f'vsx_{self.var_type}_approxed'))

    def to_csv(self, path, n_approx=128, max_egr=np.inf):
        if os.path.isdir(path):
            path = os.path.join(path, f'{self.var_type}.csv.bz2')
        with numpy_print_options(threshold=sys.maxsize):
            self.with_approxed(n_approx, endpoint=False, max_egr=max_egr).to_csv(path)

    @property
    def _lclib_model_name(self):
        t = self.var_type.upper()
        t = greek_to_latin(t)
        t = re.sub(r'[^A-Z0-9]', '', t)
        return f'ZTFDR3VSX{t}'

    def to_lclib(self, path, n_obj=100, n_approx=128, max_egr=np.inf, survey='ZTF', rng=None):
        logging.info(f'Generating {self.var_type} LCLIB for survey {survey}')

        rng = np.random.default_rng(rng)

        if self.var_type != 'Cepheid':
            logging.warning('LCLIB supports Cepheid variable type only')
            return

        if survey == 'ZTF':
            def magn(row, i, band):
                return row[self.model_column(band)][i]

            bands = self.bands
            band_names = [BAND_NAMES[band] for band in bands]
        elif survey == 'LSST':
            def magn(row, i, band):
                return row[self.lsst_model_column(band)][i]

            bands = LSST_BANDS
            band_names = LSST_BANDS
        else:
            raise ValueError(f'survey={survey} is not supported, use ZTF or LSST')

        if os.path.isdir(path):
            path = os.path.join(path, f'{survey}_{self.var_type}.lclib')

        df = self.with_approxed(n_approx, endpoint=True, max_egr=max_egr)
        model = CepheidModel(df, self.data.dustmaps_cache_dir)

        with open(path, 'w') as fh:
            fh.write(
                f'SURVEY: {survey}\n'
                f'FILTERS: {band_names}\n'
                f'MODEL: {self._lclib_model_name}\n'  # CHECKME
                'MODEL_PARNAMES: VSXOID,PERIOD\n'  # CHECKME
                'RECUR_CLASS: RECUR-PERIODIC\n'
                f'NEVENT: {n_obj}\n'
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
            for i_event in range(n_obj):
                row = model.sample(rng=rng)
                fh.write(
                    f'START_EVENT: {i_event}\n'
                    f'NROW: {n_approx} RA: {row.ra} DEC: {row.dec}\n'
                    f'PARVAL: {row.name},{row.period:.6g}\n'
                    # This option could lead to infinite loop if there is a sky position in a survey not covered by any
                    # LCLIB entry
                    # f'ANGLEMATCH_b: {self.max_abs_b}\n'  # CHECKME
                )
                for i in range(n_approx):
                    time = row.folded_time_model[i]
                    fh.write(f'S: {time:7.4f}')
                    for band in bands:
                        fh.write(f' {magn(row, i, band):.3f}')
                    fh.write('\n')
                fh.write(
                    f'END_EVENT: {i_event}\n'
                    '\n'
                )


class CepheidModel:
    period_lognorm_s = 1.0
    period_min = 0.3
    ln_period_min = np.log(period_min)
    period_max = 300
    ln_period_max = np.log(period_max)

    def __init__(self, df, dustmaps_cache_dir):
        self.df = df
        self.mw_density = MilkyWayDensityJuric2008()
        self.periods = self.df['period'].to_numpy(dtype=np.float)
        self.dustmaps_cache_dir = dustmaps_cache_dir

    def model_period_pdf(self, period, mean_period):
        scale = mean_period / stats.lognorm.mean(s=self.period_lognorm_s)
        return stats.lognorm.pdf(period, s=self.period_lognorm_s, scale=scale)

    def Mv_period(self, period):
        # https://arxiv.org/pdf/0707.3144.pdf page 18
        Mv = -3.932 - 2.819 * (np.log10(period) - 1.0)
        return Mv

    def sample(self, rng=None):
        rng = np.random.default_rng(rng)
        period = self.sample_period(rng=rng)
        Mv = self.Mv_period(period)

        # Chose random object prototype with close period
        prob = self.model_period_pdf(period, self.periods)
        row = self.df.sample(n=1, weights=prob, random_state=rng._bit_generator).iloc[0]

        # Get random coordinates
        coords = self.mw_density.sample_eq(rng=rng)
        # Magnitude correction to move object to given distance, assuming g is V
        mv = Mv + coords.distance.distmod.value
        dm = mv - row['mag_folded_model_g'].mean()

        # Get extinction
        row['ebv'] = get_sfd_thin_disk_ebv(coords.ra.deg, coords.dec.deg, coords.distance.pc, self.dustmaps_cache_dir)

        # Change period
        row['folded_time_model'] *= period / row['period']
        row['period'] = period

        # Apply random color shift and add extinction
        for column in row.keys():
            if not column.startswith('mag_folded_model_'):
                continue
            band = column[-1]
            # Apply dm, random color shift and extinction
            row[column] += dm + row['ebv'] * LSST_A_TO_EBV[band] + rng.normal(scale=0.03)

        # Set coordinates
        row['ra'] = coords.ra.to_value(u.deg)
        row['dec'] = coords.dec.to_value(u.deg)
        row['distance_pc'] = coords.distance.to_value(u.pc)

        return row

    def sample_period(self, shape=(), rng=None):
        rng = np.random.default_rng(rng)
        # Uniform log-period
        period = np.exp(rng.uniform(low=self.ln_period_min, high=self.ln_period_max, size=shape))
        return period


def prepare_vsx_folded(cli_args):
    import ch_vars.data.good_vsx_folded as data

    good_ids = get_ids(data)
    data_getter = VsxDataGetter(cli_args.data, cli_args.cache, good_ids)
    for var_type, ids in good_ids.items():
        model = VsxFoldedModel(data_getter, var_type, ids)
        if cli_args.csv:
            model.to_csv(cli_args.output, max_egr=cli_args.maxegr)
        if cli_args.lclib:
            model.to_lclib(cli_args.output, max_egr=cli_args.maxegr, survey='ZTF', rng=0)
            model.to_lclib(cli_args.output, max_egr=cli_args.maxegr, survey='LSST', rng=0)
        if cli_args.plots:
            model.plots(cli_args.output)


def parse_args():
    parser = ArgumentParser('Create Gaussian Process approximated models')
    parser.add_argument('--csv', action='store_true', help='save into .csv.bz2 file')
    parser.add_argument('--lclib', action='store_true', help='save in SNANA LCLIB format')
    parser.add_argument('--plots', action='store_true', help='plot figures')
    parser.add_argument('-d', '--data', default='https://static.rubin.science/',
                        help='data root, could be local path or HTTP URL (URL IS BROKEN FOR VSX DUE TO AN ISSUE WITH PANDAS)')
    parser.add_argument('--cache', default=None, help='directory to use as cache location')
    parser.add_argument('-o', '--output', default='.', help='directory to save models')
    parser.add_argument('--maxegr', default=np.inf, type=float,
                        help='filter objects with E_gr larger or equal than given')
    args = parser.parse_args()
    return args


def main():
    logging.basicConfig(level=logging.DEBUG)
    cli_args = parse_args()
    prepare_vsx_folded(cli_args)


if __name__ == '__main__':
    main()
