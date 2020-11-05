from copy import copy
from typing import Callable, Optional

import numpy as np
from gatspy.periodic import LombScargleMultiband, LombScargleMultibandFast
from george import GP
from george.kernels import ExpSquaredKernel, ExpSine2Kernel


def fold_lc(obj, period_range):
    if obj['mjd'].size < 64:
        lsmf = LombScargleMultiband(fit_period=True)
    else:
        lsmf = LombScargleMultibandFast(fit_period=True)
    period_range = period_range[0], min(period_range[1], np.ptp(obj['mjd']))
    lsmf.optimizer.period_range = period_range
    lsmf.fit(obj['mjd'], obj['mag'], obj['magerr'], obj['filter'])
    period = lsmf.best_period
    phase = obj['mjd'] % period / period
    records = copy(obj.item()) + (phase, period,)
    dtype = obj.dtype.descr + [('phase', object), ('period', float)]
    folded = np.rec.fromrecords(records, dtype=dtype)
    return folded


def approx(lc, timescale=30) -> Optional[Callable]:
    if lc['mjd'].size < 10:
        return None
    flux = np.power(10.0, -0.4 * lc['mag'])
    fluxerr = 0.5 * (
        np.power(10.0, -0.4 * (lc['mag'] - lc['magerr']))
        - np.power(10.0, -0.4 * (lc['mag'] + lc['magerr']))
    )
    kernel = (10.0 * np.max(flux))**2 * ExpSquaredKernel(timescale)
    gp = GP(kernel)
    gp.compute(lc['mjd'], fluxerr)

    def f(x):
        return -2.5 * np.log10(gp.predict(flux, x, return_cov=False, return_var=False))

    return f


def approx_periodic(lc, period=1.0) -> Optional[Callable]:
    _, idx = np.unique(lc['phase'], return_index=True)
    if idx.size < 10:
        return None
    phase = lc['phase'][idx]
    mag = lc['mag'][idx]
    magerr = lc['magerr'][idx]
    mean = np.average(mag, weights=magerr**-2)
    mag -= mean
    kernel = ExpSine2Kernel(gamma=1.0, log_period=np.log(period))
    # No fitting => no reason to freeze
    # kernel.freeze_parameter('k2:log_period')
    gp = GP(kernel)
    gp.compute(phase, magerr)

    def f(x):
        return gp.predict(mag, x, return_cov=False, return_var=False) + mean

    return f


__all__ = ('fold_lc', 'approx', 'approx_periodic',)
