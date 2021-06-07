import importlib.resources

import astropy
import astropy.units as u
import numpy as np
from astropy.coordinates import Angle, Distance, SkyCoord
from astropy.table import QTable

from ch_vars.data import open_ngc
from ch_vars.data.hyperleda import get_hyperleda


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
