import logging
import os
from abc import ABC, abstractmethod
from argparse import ArgumentParser

import astropy.units as u
import numpy as np
from astropy.coordinates import Angle, Distance, SkyCoord, Galactocentric, ICRS

from ch_vars.common import nearest
from ch_vars.data import get_ngc_galaxies


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
    ages_table2 = np.array([0.15, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0]) * u.Gyr
    dens0_thin_disk_table2 = np.array([4.0, 7.9, 6.2, 4.0, 5.8, 4.9, 6.6]) * 1e-3 * u.Msun / u.pc ** 3
    dens0_thick_disk = 1.34e-3 * u.Msun / u.pc ** 3
    dens0_halo = 9.32e-6 * u.Msun / u.pc ** 3
    epsilon_thin_disk_table2 = np.array([0.0140, 0.0268, 0.0375, 0.0551, 0.0696, 0.785, 0.791])
    epsilon_halo = 0.76

    # Robin et al., 2012, Table 1, Model B
    # https://doi.org/10.1051/0004-6361/201116512
    phi_bulge = np.deg2rad(11.10)
    x0_bulge_kpc = 4.07
    y0_bulge_kpc = 0.76
    z0_bulge_kpc = 0.41
    bulge_dens_norm_unweighted = 23.83e9 * u.Msun / u.pc ** 3
    r_c_bulge_kpc = 5.99
    c_parallel_bulge = 1.434
    c_perpendicular_bulge = 3.797

    @property
    def dens0_thin_disk(self):
        # return self.dens0_thin_disk_table2[np.searchsorted(self.ages_table2, self.age)]
        return np.sum(self.dens0_thin_disk_table2)

    @property
    def epsilon_thin_disk(self):
        return self.epsilon_thin_disk_table2[np.searchsorted(self.ages_table2, self.age)]

    def thin_disk_dens(self, rho, phi, z):
        a = np.hypot(rho, z / self.epsilon_thin_disk)

        if self.age <= 0.15 * u.Gyr:
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
            self.thin_disk_dens_norm * self.thin_disk_dens(rho, phi, z)
            + self.thick_disk_dens_norm * self.thick_disk_dens(rho, phi, z)
            + self.halo_dens_norm * self.halo_dens(rho, phi, z)
            + self.bulge_dens_norm * self.bulge_dens(rho, phi, z)
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


def galactic_density_from_ngc_row(row, milky_way):
    hubble_type = row['Hubble']
    if np.ma.is_masked(hubble_type):
        raise ValueError('No Hubble type')
    if hubble_type.startswith('S'):
        return SpiralGalaxyDensity.from_ngc_row(row, milky_way=milky_way)
    if hubble_type.startswith('E'):
        return EllipticalGalaxyDensity.from_ngc_row(row)
    raise ValueError(f'Unsupported galaxy type: {hubble_type}')


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

    # density = MilkyWayDensityBesancon(age=30*u.Myr, bulge_weight=0.0)
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
