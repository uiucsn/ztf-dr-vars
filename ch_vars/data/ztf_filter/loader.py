import importlib.resources
from functools import lru_cache

import numpy as np
from scipy.integrate import simps

from ch_vars.common import BAND_NAMES, mean_reduce
from ch_vars.data import ztf_filter


def fix_negative(a):
    index = np.arange(a.size, dtype=a.dtype)
    i = a < 0
    a[i] = np.interp(index[i], index[~i], a[~i])


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
