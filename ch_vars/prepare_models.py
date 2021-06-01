import importlib.resources
import logging
import os
import re
import sys
from abc import ABC, abstractmethod
from argparse import ArgumentParser
from datetime import datetime
from functools import lru_cache, partial

import astropy.constants as const
import astropy.io.ascii
import astropy.table
import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import Angle, Distance, SkyCoord, Galactocentric, ICRS
from astropy.table import QTable
from astroquery.vizier import Vizier
from joblib import Memory
from pyvo.dal import TAPService
from scipy.integrate import simps
from scipy.optimize import least_squares
from scipy import stats

from ch_vars.approx import approx_periodic, fold_lc
from ch_vars.catalogs import CATALOGS
from ch_vars.common import BAND_NAMES, COLORS, greek_to_latin, str_to_array, numpy_print_options, LSST_BANDS,\
    LSST_COLORS, nearest
from ch_vars.data import hyperleda, lsst_filter, open_ngc, ztf_filter
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


class _DustMap:
    bayestar_version = 'bayestar2019'
    bayestar_r = {1: 3.518, 2: 2.617, 3: 1.971}

    def __init__(self, cache_dir):
        from dustmaps import bayestar

        if cache_dir is not None:
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


def fix_negative(a):
    index = np.arange(a.size, dtype=a.dtype)
    i = a < 0
    a[i] = np.interp(index[i], index[~i], a[~i])


def mean_reduce(a, factor=10):
    if a.size % factor != 0:
        a = np.concatenate([a, np.full(factor - a.size % factor, a[-1])])
    mean = a.reshape(-1, factor).mean(axis=1)
    return mean


@lru_cache()
def ztf_band_transmission(band: int):
    filename = f'ztf_{BAND_NAMES[band]}_band.csv'
    with importlib.resources.open_binary(ztf_filter, filename) as fh:
        lmbd, r = np.genfromtxt(fh, delimiter=',', unpack=True)
    if lmbd[0] > lmbd[-1]:
        lmbd = lmbd[::-1]
        r = r[::-1]
    lmbd *= 1e-7  # from nm to cm
    r *= 1e-2  # from percent to fraction

    fix_negative(r)

    lmbd = mean_reduce(lmbd)
    r = mean_reduce(r)
    norm = simps(x=lmbd, y=r)
    return lmbd, r, norm


@lru_cache()
def lsst_band_transmission(band: str):
    filename = f'lsst_total_{band.lower()}.dat'
    with importlib.resources.open_binary(lsst_filter, filename) as fh:
        lmbd, r = np.genfromtxt(fh, delimiter=' ', comments='#', unpack=True)

    lmbd *= 1e-7  # from nm to cm

    lmbd = mean_reduce(lmbd)
    r = mean_reduce(r)

    norm = simps(x=lmbd, y=r)
    return lmbd, r, norm


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


class MilkyWayDensityBase(ABC):
    sun_z_kpc = None
    sun_rho_kpc = None

    rho_min_kpc = 0
    rho_max_kpc = 20
    z_min_kpc = -20
    z_max_kpc = 20

    def __init__(self, n_grid=2048):
        self.n_grid = n_grid

        self.rho, self.z = np.meshgrid(
            np.linspace(self.rho_max_kpc, self.rho_min_kpc, n_grid),
            np.linspace(self.z_min_kpc, self.z_max_kpc, n_grid)
        )
        self.dens_grid = self.dens(self.rho, 0.0, self.z)
        self.dens_flat_cum = np.cumsum(self.dens_grid)
        self.dens_flat_cum /= self.dens_flat_cum[-1]

    @abstractmethod
    def dens(self, rho, phi, z):
        raise NotImplemented

    @staticmethod
    def xyz_from_cyl(rho, phi, z):
        x = rho * np.cos(phi)
        y = rho * np.sin(phi)
        return x, y, z

    def sample_gal_xyz(self, shape=(), rng=None):
        rho, phi, z = self.sample_gal_cyl(shape=shape, rng=rng)
        return self.xyz_from_cyl(rho, phi, z)

    def eq_from_gal_cyl(self, rho, phi, z):
        x = np.cos(phi) * rho
        y = np.sin(phi) * rho
        gc = Galactocentric(x=x*u.kpc, y=y*u.kpc, z=z*u.kpc,
                            z_sun=self.sun_z_kpc*u.kpc, galcen_distance=self.sun_rho_kpc*u.kpc)
        eq = gc.transform_to(ICRS())
        return eq

    def sample_eq(self, shape=(), rng=None):
        return self.eq_from_gal_cyl(*self.sample_gal_cyl(shape=shape, rng=rng))

    def sample_gal_cyl(self, shape=(), rng=None):
        random_rng = np.random.default_rng(rng)
        r = random_rng.random(size=shape)
        idx = nearest(self.dens_flat_cum, r)
        rho = self.rho.reshape(-1)[idx]
        z = self.z.reshape(-1)[idx]
        phi = random_rng.uniform(0, 2*np.pi, size=shape)
        return rho, phi, z


class MilkyWayDensityBesancon(MilkyWayDensityBase):
    """Robin et al, 2003, mainly Table 3
    https://doi.org/10.1051/0004-6361:20031117
    """
    sun_rho_kpc = 8.5
    sun_z_kpc = 0.015

    def __init__(self, age, n_grid=2048,
                 thin_disk_weight=1.0, thick_disk_weight=1.0, halo_weight=1.0, bulge_weight=1.0):
        self.age = age

        self.thin_disk_dens_norm = thin_disk_weight * self._dens_norm_from_local(self.thin_disk_dens,
                                                                                 self.dens0_thin_disk)
        self.thick_disk_dens_norm = thick_disk_weight * self._dens_norm_from_local(self.thick_disk_dens,
                                                                                   self.dens0_thick_disk)
        self.halo_dens_norm = halo_weight * self._dens_norm_from_local(self.halo_dens, self.dens0_halo)
        self.bulge_dens_norm = bulge_weight * self.bulge_dens_norm_unweighted

        super().__init__(n_grid=n_grid)

    def _dens_norm_from_local(self, dens_cyl_func, dens_local):
        return dens_local / dens_cyl_func(self.sun_rho_kpc, 0.0, self.sun_z_kpc)

    # Robin et al, 2003, Table 2
    ages_table2 = np.array([0.15, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0]) * u.Myr
    dens0_thin_disk_table2 = np.array([4.0, 7.9, 6.2, 4.0, 5.8, 4.9, 6.6]) * 1e-3 * u.Msun / u.pc ** 3
    dens0_thick_disk = 1.34e-3 * u.Msun / u.pc ** 3
    dens0_halo = 9.32e-6 * u.Msun / u.pc ** 3
    epsilon_thin_disk_table2 = np.array([0.0140, 0.0268, 0.0375, 0.0551, 0.0696, 0.785, 0.791])
    epsilon_halo = 0.76

    # Robin et al., 2012, Table 1, Model B
    # https://doi.org/10.1051/0004-6361/201116512
    phi_bulge = 11.10 * u.deg
    x0_bulge_kpc = 4.07
    y0_bulge_kpc = 0.76
    z0_bulge_kpc = 0.41
    bulge_dens_norm_unweighted = 23.83e9 * u.Msun / u.pc ** 3
    r_c_bulge_kpc = 5.99
    c_parallel_bulge = 1.434
    c_perpendicular_bulge = 3.797

    @property
    def dens0_thin_disk(self):
        return self.dens0_thin_disk_table2[np.searchsorted(self.ages_table2, self.age)]

    @property
    def epsilon_thin_disk(self):
        return self.epsilon_thin_disk_table2[np.searchsorted(self.ages_table2, self.age)]

    def thin_disk_dens(self, rho, phi, z):
        a = np.hypot(rho, z / self.epsilon_thin_disk)

        if self.age <= 0.15 * u.Myr:
            h_r_plus_kpc = 5.0
            h_r_minus_kpc = 3.0
            return np.exp(-(a / h_r_plus_kpc) ** 2) - np.exp(-(a / h_r_minus_kpc) ** 2)

        # Robin et al., 2012, Table 1, Model B, h_r_plus is R_d (aka disk) and h_r_minus is R_h (aka hole)
        h_r_plus_kpc = 2.26
        h_r_minus_kpc = 0.18
        return np.exp(-np.hypot(0.5, a / h_r_plus_kpc)) - np.exp(-np.hypot(0.5, a / h_r_minus_kpc))

    def thick_disk_dens(self, rho, phi, z):
        x_l_kpc = 0.4
        h_r_kpc = 2.5
        h_z_kpc = 0.8

        dens = (
            np.exp(-(rho - self.sun_rho_kpc) / h_r_kpc)
            * np.where(
                np.abs(z) <= x_l_kpc,
                (1.0 - z**2 / (h_z_kpc * x_l_kpc * (2.0 + x_l_kpc / h_z_kpc))),
                np.exp(x_l_kpc / h_z_kpc) / (1.0 + 0.5 * x_l_kpc / h_z_kpc) * np.exp(-np.abs(z) / h_z_kpc),
            )
        )
        return dens

    def halo_dens(self, rho, phi, z):
        a = np.hypot(rho, z / self.epsilon_halo)
        a_c_kpc = 0.5
        return np.power(
            np.where(a <= a_c_kpc, a_c_kpc, a) / self.sun_rho_kpc,
            -2.44,
        )

    def bulge_dens(self, rho, phi, z):
        """From Robin et al., 2012, model B

        https://doi.org/10.1051/0004-6361/201116512
        """
        rho_bulge, phi_bulge, z_bulge = rho, phi - self.phi_bulge, z
        x_abs_bulge, y_abs_bulge, z_abs_bulge = (np.abs(v) for v in self.xyz_from_cyl(rho_bulge, phi_bulge, z_bulge))
        r_s_kpc = np.power(
            np.power(
                np.power(x_abs_bulge/self.x0_bulge_kpc, self.c_perpendicular_bulge)
                + np.power(y_abs_bulge / self.y0_bulge_kpc, self.c_perpendicular_bulge),
                self.c_parallel_bulge / self.c_perpendicular_bulge,
            )
            + np.power(
                z_abs_bulge/self.z0_bulge_kpc,
                self.c_parallel_bulge,
            ),
            1.0 / self.c_parallel_bulge,
        )
        dens_s = 1.0 / np.cosh(-r_s_kpc) ** 2
        cut_off_f_c = np.where(
            rho_bulge <= self.r_c_bulge_kpc,
            1.0,
            np.exp(-((rho_bulge - self.r_c_bulge_kpc) / 0.5)**2)
        )
        return dens_s * cut_off_f_c

    def dens(self, rho, phi, z):
        mass_dens = (
            self.thin_disk_dens(rho, phi, z)
            + self.thick_disk_dens(rho, phi, z)
            + self.halo_dens(rho, phi, z)
            + self.bulge_dens(rho, phi, z)
        )
        return mass_dens.to_value(u.Msun / u.pc**3)


class MilkyWayDensityJuric2008(MilkyWayDensityBase):
    """Juric et al, 2008, Table 10
    https://iopscience.iop.org/article/10.1086/523619/pdf
    """
    sun_rho_kpc = 8
    sun_z_kpc = 0.024
    l_thin_kpc = 2.6
    h_thin_kpc = 0.3
    dens_thick_to_thin = 0.12
    l_thick_kpc = 3.6
    h_thick_kpc = 0.9
    dens_halo_to_thin = 0.0051
    ellipticity_halo = 0.64
    power_order_halo = 2.77

    rho_min_kpc = 1
    rho_max_kpc = 20
    z_min_kpc = -10
    z_max_kpc = 10

    def __init__(self, n_grid=2048, thin_disk_weight=1.0, thick_disk_weight=1.0, halo_weight=1.0):
        self.thin_disk_weight = thin_disk_weight
        self.thick_disk_weight = thick_disk_weight
        self.halo_weight = halo_weight

        super().__init__(n_grid=n_grid)

    def _disk_dens(self, rho, phi, z, l, h):
        return np.exp((self.sun_rho_kpc - rho) / l - np.abs(z) / h)

    def thin_disk_dens(self, rho, phi, z):
        return self._disk_dens(rho, phi, z, l=self.l_thin_kpc, h=self.h_thin_kpc)

    def thick_disk_dens(self, rho, phi, z):
        return self.dens_thick_to_thin * self._disk_dens(rho, phi, z, l=self.l_thick_kpc, h=self.h_thick_kpc)

    def halo_dens(self, rho, phi, z):
        return (self.dens_halo_to_thin
                * np.power(self.sun_rho_kpc / np.hypot(rho, z / self.ellipticity_halo), self.power_order_halo))

    def dens(self, rho, phi, z):
        return (
                self.thin_disk_weight * self.thin_disk_dens(rho, phi, z)
                + self.thick_disk_weight * self.thick_disk_dens(rho, phi, z)
                + self.halo_weight * self.halo_dens(rho, phi, z)
        )


class MilkyWayLikeGalaxyDensity:
    def __init__(self, size_scale, milky_way):
        self.size_scale = size_scale
        self.mw = milky_way

    def sample_gal_cyl(self, shape=(), rng=None):
        rho_phi_z = tuple(x * self.size_scale for x in self.mw.sample_gal_cyl(shape=shape, rng=rng))
        return rho_phi_z

    def sample_gal_xyz(self, shape=(), rng=None):
        xyz = tuple(a * self.size_scale for a in self.mw.sample_gal_xyz(shape=shape, rng=rng))
        return xyz


class ExtragalacticDensity(ABC):
    def __init__(self, center_coord: SkyCoord, position_angle_ne: Angle, major_axis: Angle, minor_axis: Angle):
        self.center_coord = center_coord
        self.position_angle_ne = position_angle_ne
        self.major_axis = major_axis
        self.minor_axis = minor_axis
        self.major_arcsec = self.major_axis.to_value(u.arcsec)
        self.minor_arcsec = self.minor_axis.to_value(u.arcsec)

    @classmethod
    def from_ngc_row(cls, row, **kwargs):
        return cls(
            center_coord=row['coord'],
            position_angle_ne=row['PosAng'],
            major_axis=row['MajAx'],
            minor_axis=row['MinAx'],
            **kwargs
        )

    @staticmethod
    def offset_coords(coords: SkyCoord, offset: Angle, rotate_ne: Angle = Angle(0*u.deg)):
        z = offset[..., 0] + offset[..., 1] * 1j
        separation = np.abs(z)
        position_angle = rotate_ne + np.angle(z)
        return coords.directional_offset_by(position_angle, separation)

    def sample_eq(self, shape=(), rng=None):
        rel_coords = self.sample_sky_relative_ellipse(shape=shape, rng=rng)
        return self.offset_coords(self.center_coord, rel_coords, rotate_ne=self.position_angle_ne)

    @abstractmethod
    def sample_sky_relative_ellipse(self, shape=(), rng=None) -> Angle:
        pass


class SpiralGalaxyDensity(ExtragalacticDensity):
    # TODO: check
    # TODO: see Burstein et al., 1982, about Sc galaxy mass distribution
    # https://ui.adsabs.harvard.edu/abs/1982ApJ...253...70B
    mw_isophote_25_kpc = 20.0

    def __init__(self, center_coord: SkyCoord, position_angle_ne: Angle, major_axis: Angle, minor_axis: Angle,
                 milky_way: MilkyWayDensityBase):
        super().__init__(center_coord, position_angle_ne, major_axis, minor_axis)

        major_axis_kpc = (2.0 * np.tan(major_axis) * center_coord.distance).to_value(u.kpc)
        size_scale = major_axis_kpc / self.mw_isophote_25_kpc
        self.milky_way_like = MilkyWayLikeGalaxyDensity(size_scale=size_scale, milky_way=milky_way)

    def sample_sky_relative_ellipse(self, shape=(), rng=None):
        x_kpc, y_kpc, z_kpc = self.milky_way_like.sample_gal_xyz(shape=shape, rng=rng)
        x_sky = x_kpc / (self.mw_isophote_25_kpc * self.milky_way_like.size_scale) * self.major_axis
        y_sky = y_kpc / (self.mw_isophote_25_kpc * self.milky_way_like.size_scale) * self.minor_axis
        return Angle(np.stack([x_sky, y_sky], axis=-1))


class EllipticalGalaxyDensity(ExtragalacticDensity):
    scale_radii_in_25_isophote = 10.0

    def sample_sky_relative_ellipse(self, shape=(), rng=None):
        if isinstance(shape, int):
            shape = (shape,)

        rng = np.random.default_rng(rng)

        angles_arcsec = rng.normal(
            loc=(0.0, 0.0),
            scale=(self.major_arcsec / self.scale_radii_in_25_isophote,
                   self.minor_arcsec / self.scale_radii_in_25_isophote),
            size=tuple(shape) + (2, )
        )
        return angles_arcsec * u.arcsec


def get_ngc_galaxies(leda_distance=True):
    """openNGC galaxies

    https://github.com/mattiaverga/OpenNGC
    See axis description here:
    https://cdsarc.unistra.fr/viz-bin/ReadMe/VII/119?format=html&tex=true

    Arguments
    ---------
    leda_distance : bool, optional
        Add distance modulus from HyperLeda and keep objects found there only

    """
    with importlib.resources.open_binary(open_ngc, 'NGC.csv') as fh:
        table = astropy.io.ascii.read(fh, format='csv', delimiter=';')
    table = table[table['Type'] == 'G']

    table['RA'] = Angle(table['RA'], unit=u.hour)
    table['Dec'] = Angle(table['Dec'], unit=u.deg)

    # 25mag/arcsec^2 isophote ellipsoidal axes
    table['MajAx'] = table['MajAx'] * u.arcmin
    table['MinAx'] = table['MinAx'] * u.arcmin

    # NE positional angle
    table['PosAng'] = table['PosAng'] * u.deg

    table['coord'] = SkyCoord(table['RA'], table['Dec'])

    table.add_index('Name')

    table = QTable(table)

    if leda_distance:
        leda = get_hyperleda()
        table = table[np.isin(table['Name'], leda.index)]
        table['distance'] = Distance(distmod=leda['modulus'].loc[table['Name']])

    return table


def get_hyperleda():
    column_names = ['objname', 'j2000', 'modulus']
    with importlib.resources.open_binary(hyperleda, 'HyperLeda_a007_1620838730.txt') as fh:
        df = pd.read_csv(
            fh,
            skiprows=17,
            comment='#',
            delimiter=r'\s+',
            names=column_names,
            usecols=column_names,
            index_col='objname',
        )
    grouped_df = pd.DataFrame(
        [{'objname': objname, 'j2000': df['j2000'].iloc[0], 'modulus': df['modulus'].median()}
         for objname, df in df.groupby(by='objname', axis=0, as_index=True)],
    )
    grouped_df = grouped_df.set_index('objname')
    return grouped_df


def galactic_density_from_ngc_row(row, milky_way):
    hubble_type = row['Hubble']
    if np.ma.is_masked(hubble_type):
        raise ValueError('No Hubble type')
    if hubble_type.startswith('S'):
        return SpiralGalaxyDensity.from_ngc_row(row, milky_way=milky_way)
    if hubble_type.startswith('E'):
        return EllipticalGalaxyDensity.from_ngc_row(row)
    raise ValueError(f'Unsupported galaxy type: {hubble_type}')


class VsxDataGetter:
    vizier_id = 'OID'

    def __init__(self, data_dir, cache_dir, type_ids):
        self.cat = CATALOGS['vsx']
        self.memory = Memory(location=cache_dir)
        if cache_dir is None:
            dustmaps_cache_dir = None
        else:
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
        model = CepheidModel(df)

        with open(path, 'w') as fh:
            fh.write(
                f'SURVEY: {survey}\n'
                f'FILTERS: {band_names}\n'
                f'MODEL: {self._lclib_model_name}\n'  # CHECKME
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

    def __init__(self, df):
        self.df = df
        self.mw_density = MilkyWayDensityJuric2008()
        self.periods = self.df['period'].to_numpy(dtype=np.float)

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
        mv = Mv + coords.distance.distmod.value
        # Magnitude correction to move object to given distance, assuming g is V
        dm = mv - row['mag_folded_model_g'].mean()

        # Change period
        row['folded_time_model'] *= period / row['period']
        row['period'] = period

        # Move object to given distance and apply random color shift
        for column in row.keys():
            if not column.startswith('mag_folded_model_'):
                continue
            # Apply dm and random color shift
            row[column] += dm + rng.normal(scale=0.03)

        # Set coordinates
        row['ra'] = coords.ra.to_value(u.deg)
        row['dec'] = coords.dec.to_value(u.deg)

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


def plot_milky_way_entrypoint(args=None):
    import matplotlib.pyplot as plt

    logging.basicConfig(level=logging.DEBUG)

    parser = ArgumentParser('Sample Milky Way distribution')
    parser.add_argument('-n', '--count', type=int, default=2000,
                        help='number of objects to generate')
    parser.add_argument('--random-seed', type=int, default=None,
                        help='random seed')
    parser.add_argument('-o', '--output', default='.',
                        help='directory to save a figure')
    cli_args = parser.parse_args(args)

    density = MilkyWayDensityJuric2008()
    coords = density.sample_eq(shape=cli_args.count, rng=cli_args.random_seed)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="mollweide")
    ax.set_title(f'Eq coordinates, n = {cli_args.count}')
    # Minus RA is a trick to have "inside" view to the celestial sphere
    scatter = ax.scatter(-coords.ra.wrap_at(180 * u.deg).radian, coords.dec.wrap_at(180 * u.deg).radian,
                         s=3, c=coords.distance.to_value(u.kpc), cmap='YlGn', vmin=0)
    ax.set_xticklabels(['10h', '8h', '6h', '4h', '2h', '0h', '22h', '20h', '18h', '16h', '14h'])
    colorbar = fig.colorbar(scatter)
    colorbar.set_label('distance, kpc')
    fig.savefig(os.path.join(cli_args.output, f'milky_way_plot_{cli_args.count}.png'))


def plot_extragalactic_entrypoint(args=None):
    import matplotlib.pyplot as plt
    from tqdm import tqdm

    logging.basicConfig(level=logging.DEBUG)

    parser = ArgumentParser('Sample extragalactic distribution')
    parser.add_argument('-g', '--name', default=None,
                        help='galaxy name, for example NGC0224 is M31; default is too plot all galaxies')
    parser.add_argument('-n', '--count', type=int, default=100,
                        help='number of objects to generate')
    parser.add_argument('--random-seed', type=int, default=None,
                        help='random seed')
    parser.add_argument('-o', '--output', default='.',
                        help='directory to save a figure')
    cli_args = parser.parse_args(args)

    ngc = get_ngc_galaxies()
    rows = [ngc.loc[cli_args.name]] if cli_args.name is not None else ngc

    milky_way_like = MilkyWayLikeGalaxyDensity(size_scale=1.0, milky_way=MilkyWayDensityJuric2008())

    fig = plt.figure(figsize=(24, 36))

    ax_proj = fig.add_subplot(211, projection="mollweide")
    ax_rect = fig.add_subplot(212)

    for galaxy in tqdm(rows):
        try:
            density = galactic_density_from_ngc_row(galaxy, milky_way_like)
        except ValueError as e:
            continue

        coords = density.sample_eq(shape=cli_args.count, rng=cli_args.random_seed)
        ax_proj.set_title(f'Eq coordinates, n = {cli_args.count}')

        # Minus RA is a trick to have "inside" view to the celestial sphere
        ax_proj.scatter(-coords.ra.wrap_at(180 * u.deg).radian, coords.dec.wrap_at(180 * u.deg).radian, s=2)
        ax_proj.set_xticklabels(['10h', '8h', '6h', '4h', '2h', '0h', '22h', '20h', '18h', '16h', '14h'])

    ax_rect.set_title(galaxy['Name'])
    ax_rect.scatter(coords.ra.wrap_at(180*u.deg).to_value(u.deg), coords.dec.to_value(u.deg))
    ax_rect.invert_xaxis()
    ax_rect.set_xlabel('RA')
    ax_rect.set_ylabel('Dec')

    fig.savefig(os.path.join(cli_args.output, f'extragal_plot_{cli_args.name}_{cli_args.count}.png'))


def plot_map_entrypoint(args=None):
    from itertools import chain, cycle

    import matplotlib.pyplot as plt
    from matplotlib.colors import TABLEAU_COLORS
    from tqdm import tqdm

    colors = list(TABLEAU_COLORS.values())

    logging.basicConfig(level=logging.DEBUG)

    parser = ArgumentParser('Sample object distribution')
    parser.add_argument('--count-mw', type=int, default=2000,
                        help='number of Milky Way objects to generate')
    parser.add_argument('--count-galaxy', type=int, default=200,
                        help='number of obects per galaxy to generate')
    parser.add_argument('-d', '--max-distance', type=float, default=50.0,
                        help='maximum distance to galaxies, Mpc')
    parser.add_argument('--random-seed', type=int, default=None,
                        help='random seed')
    parser.add_argument('-o', '--output', default='.',
                        help='directory to save a figure')
    cli_args = parser.parse_args(args)

    rng = np.random.default_rng(cli_args.random_seed)

    milky_way = MilkyWayDensityJuric2008()
    mw_coords = milky_way.sample_eq(shape=cli_args.count_mw, rng=rng)

    ngc = get_ngc_galaxies()
    ngc = ngc[ngc['distance'] <= Distance(cli_args.max_distance, unit=u.Mpc)]
    milky_way_like = MilkyWayLikeGalaxyDensity(size_scale=1.0, milky_way=MilkyWayDensityJuric2008())
    extra_coords = []
    for galaxy in tqdm(ngc):
        try:
            density = galactic_density_from_ngc_row(galaxy, milky_way_like)
        except ValueError:
            continue
        extra_coords.append(density.sample_eq(shape=cli_args.count_galaxy, rng=rng))
    extra_colors = list(chain.from_iterable([color] * coords.size for color, coords in zip(cycle(colors), extra_coords)))
    extra_coords = SkyCoord(extra_coords)

    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection="mollweide")
    # Minus RA is a trick to have "inside" view to the celestial sphere
    ax.plot(-mw_coords.ra.wrap_at(180 * u.deg).radian, mw_coords.dec.wrap_at(180 * u.deg).radian,
            's', color='black', ms=3, label='MW')
    ax.scatter(-extra_coords.ra.wrap_at(180 * u.deg).radian, extra_coords.dec.wrap_at(180 * u.deg).radian,
               c=extra_colors, s=3, label='NGC')
    ax.set_xticklabels(['10h', '8h', '6h', '4h', '2h', '0h', '22h', '20h', '18h', '16h', '14h'])
    ax.legend()
    fig.savefig(os.path.join(cli_args.output, f'map_plot_{cli_args.count_mw}_{cli_args.count_galaxy}.png'))


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
