import logging
import math
import os
from argparse import ArgumentParser

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord, Galactocentric
from scipy.integrate import quad

try:
    from numba import cfunc
    from numba.types import intc, CPointer, float64
    from scipy import LowLevelCallable

    def quad_callable(f):
        f = cfunc(float64(intc, CPointer(float64)))(f)
        return LowLevelCallable(f.ctypes)
except ImportError:
    import warnings
    from functools import wraps

    warnings.warn('numba is not installed, MilkyWayExtinction and get_sfd_thin_disk_ebv will be slow', ImportWarning)

    def quad_callable(f):
        @wraps(f)
        def wrapped(x, *args):
            return f(1 + len(args), (x,) + args)
        return wrapped


class _BayestarDustMap:
    bayestar_version = 'bayestar2019'
    bayestar_r = {1: 3.518, 2: 2.617, 3: 1.971}

    def __init__(self, cache_dir):
        from dustmaps import bayestar

        if cache_dir is not None:
            bayestar.config['data_dir'] = cache_dir
        bayestar.fetch(version=self.bayestar_version)
        self.bayestar = bayestar.BayestarQuery(version=self.bayestar_version, max_samples=0)


class BayestarDustMap(_BayestarDustMap):
    _objects = {}

    def __new__(cls, cache_dir):
        if cache_dir not in cls._objects:
            cls._objects[cache_dir] = _BayestarDustMap(cache_dir)
        return cls._objects[cache_dir]


def bayestar_get(ra, dec, distance, cache_dir):
    coords = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, distance=distance*u.pc)
    return BayestarDustMap(cache_dir).bayestar(coords, mode='best')


class _SFDDustMap:
    def __init__(self, cache_dir):
        from dustmaps import sfd

        if cache_dir is not None:
            sfd.config['data_dir'] = cache_dir
        sfd.fetch()
        self.sfd = sfd.SFDQuery()


class SFDDustMap(_SFDDustMap):
    _objects = {}

    def __new__(cls, cache_dir):
        if cache_dir not in cls._objects:
            cls._objects[cache_dir] = _SFDDustMap(cache_dir)
        return cls._objects[cache_dir]


class PatchedBayestarDustMap:
    def __init__(self, cache_dir):
        self.bayestar = BayestarDustMap(cache_dir).bayestar
        self.sfd = SFDDustMap(cache_dir).sfd

    def bayestar_ext(self, coord):
        gal = coord.galactic
        flipped_coords = SkyCoord(l=-gal.l, b=-gal.b, distance=gal.distance, frame='galactic').icrs
        sfd = self.sfd(coord)
        flipped_sfd = self.sfd(flipped_coords)
        extinction = self.bayestar(coord, mode='best')
        flipped_extinction = self.bayestar(flipped_coords, mode='best')
        normalized_flipped_extinction = np.where(
            (flipped_sfd != 0) & (sfd != 0),
            flipped_extinction * sfd / flipped_sfd,
            flipped_extinction,
        )
        ext = np.where(np.isnan(extinction), normalized_flipped_extinction, extinction)
        return ext

    def ebv(self, coord):
        ext = self.bayestar_ext(coord)
        return 0.884 * ext


def get_patched_ebv(ra, dec, distance, cache_dir):
    coords = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, distance=distance * u.pc)
    return PatchedBayestarDustMap(cache_dir).ebv(coords)


@quad_callable
def milky_way_dust_density(_, xx):
    d = xx[0]
    cosl_cosb = xx[1]
    sinl_cosb = xx[2]
    sinb = xx[3]

    # This coefficient is
    # sun_galcenter_kpc = Galactocentric().galcen_distance.to_value(u.kpc)
    # math.sqrt(sun_galcenter_kpc ** 2 - z_sun_kpc ** 2)
    x = 8.1219733661223 - cosl_cosb * d
    y = sinl_cosb * d
    # Coefficient is
    # Galactocentric().z_sun.to_value(u.kpc)
    z = 0.0208 + sinb * d
    rho = math.hypot(x, y)

    # r-scale is 4 kpc, z-scale is 0.14 kpc
    return math.exp(-rho / 4.0 - math.fabs(z) / 0.14)


class MilkyWayExtinction:
    rho_0_kpc = 4.0
    z_0_kpc = 0.14

    sun_galcenter_kpc = Galactocentric().galcen_distance.to_value(u.kpc)
    z_sun_kpc = Galactocentric().z_sun.to_value(u.kpc)
    rho_sun_kpc = math.sqrt(sun_galcenter_kpc ** 2 - z_sun_kpc ** 2)

    dl_deg = 0
    db_deg = -math.degrees(math.atan2(z_sun_kpc, rho_sun_kpc))

    __slots__ = ('cosl_cosb', 'sinl_cosb', 'sinb',)

    def __init__(self, l_deg, b_deg):
        l = math.radians(l_deg + self.dl_deg)
        b = math.radians(b_deg + self.db_deg)
        self.cosl_cosb = math.cos(l) * math.cos(b)
        self.sinl_cosb = math.sin(l) * math.cos(b)
        self.sinb = math.sin(b)

    def __call__(self, d_kpc):
        return quad(milky_way_dust_density, 0.0, d_kpc, args=(self.cosl_cosb, self.sinl_cosb, self.sinb))[0]


def _get_sfd_thin_disk_ebv(sfd, l_deg, b_deg, distance_pc):
    d_kpc = 1e-3 * distance_pc
    mwdd = MilkyWayExtinction(l_deg, b_deg)
    norm = sfd / mwdd(np.inf)
    return norm * mwdd(d_kpc)


_get_sfd_thin_disk_ebv_vec = np.vectorize(_get_sfd_thin_disk_ebv)


def get_sfd_thin_disk_ebv(ra, dec, distance_pc, cache_dir):
    coord = SkyCoord(ra=ra, dec=dec, unit='deg')
    gal = coord.galactic
    sfd = SFDDustMap(cache_dir).sfd(coord)
    return _get_sfd_thin_disk_ebv_vec(sfd, gal.l.deg, gal.b.deg, distance_pc)


# https://iopscience.iop.org/article/10.1088/0004-637X/737/2/103#apj398709app1
LSST_A_TO_EBV = {
    'u': 4.145,
    'g': 3.237,
    'r': 2.273,
    'i': 1.684,
    'z': 1.323,
    'Y': 1.088,
}


def plot_extinction_entrypoint(args=None):
    import matplotlib.pyplot as plt

    logging.basicConfig(level=logging.DEBUG)

    parser = ArgumentParser('Extinction maps')
    parser.add_argument('-o', '--output', default='figs',
                        help='directory to save a figure')
    parser.add_argument('-c', '--cache', default='cache/dustmaps')
    parser.add_argument('-d', '--distance', type=float, default=1000,
                        help='distance in pc')
    cli_args = parser.parse_args(args)

    ra, dec = np.meshgrid(np.linspace(-180, 180, 361, endpoint=False), np.linspace(-90, 90, 181, endpoint=False))
    coords = SkyCoord(ra=ra * u.degree, dec=dec * u.degree, distance=cli_args.distance * u.pc)

    quasi_sfd = get_sfd_thin_disk_ebv(coords.ra.deg, coords.dec.deg, coords.distance.pc, cli_args.cache)
    quasi_bayestar = PatchedBayestarDustMap(cli_args.cache).ebv(coords)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="mollweide")
    # Minus RA is a trick to have "inside" view to the celestial sphere
    scatter = ax.scatter(-coords.ra.wrap_at(180 * u.deg).radian, coords.dec.wrap_at(180 * u.deg).radian,
                         s=3, c=quasi_sfd, cmap='Greys', vmin=0, vmax=2)
    ax.set_xticklabels(['10h', '8h', '6h', '4h', '2h', '0h', '22h', '20h', '18h', '16h', '14h'])
    ax.set_title(f'Distance = {cli_args.distance} pc')
    colorbar = fig.colorbar(scatter)
    colorbar.set_label('E(B-V)')
    fig.savefig(os.path.join(cli_args.output, f'extinction_SFD_extrapolation_{cli_args.distance}pc.png'))

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="mollweide")
    # Minus RA is a trick to have "inside" view to the celestial sphere
    scatter = ax.scatter(-coords.ra.wrap_at(180 * u.deg).radian, coords.dec.wrap_at(180 * u.deg).radian,
                         s=3, c=quasi_bayestar, cmap='Greys', vmin=0, vmax=2)
    ax.set_xticklabels(['10h', '8h', '6h', '4h', '2h', '0h', '22h', '20h', '18h', '16h', '14h'])
    ax.set_title(f'Distance = {cli_args.distance} pc')
    colorbar = fig.colorbar(scatter)
    colorbar.set_label('E(B-V)')
    fig.savefig(os.path.join(cli_args.output, f'extinction_Bayestar_flipping_{cli_args.distance}pc.png'))
