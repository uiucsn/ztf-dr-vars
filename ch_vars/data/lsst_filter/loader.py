import importlib.resources
from functools import lru_cache

import numpy as np
from scipy.integrate import simps

from ch_vars.common import mean_reduce
from ch_vars.data import lsst_filter


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
