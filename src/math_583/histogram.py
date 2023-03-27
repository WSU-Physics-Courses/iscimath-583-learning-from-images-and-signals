"""Histograms with error estimates.
"""
import inspect
import warnings

import numpy as np

import scipy.stats
import scipy as sp

import matplotlib.pyplot as plt

__all__ = ["histogram_err", "hist_err"]


def hist_err(
    a,
    histtype="step",
    sigma_bounds=(-1, 1),
    bins=None,
    range=None,
    weights=None,
    density=False,
    errorbar_kw=None,
    **kw
):
    """Adds errorbars to matplotlib's hist.

    Other Parameters
    ----------------
    sigma_bounds : (float, float)
        Confidence region to plot expressed in terms of 1D normal sigma percentiles.
    errorbar_kw : dict
        Arguments for plt.errorbar()
    """
    h, dh, bins = histogram_err(
        a,
        sigma_bounds=sigma_bounds,
        bins=bins,
        range=range,
        weights=weights,
        density=density,
    )
    x = 0.5 * (bins[1:] + bins[:-1])
    stairs = plt.stairs(h, bins, **kw)
    if errorbar_kw is None:
        errorbar_kw = {}
    errorbar_kw = dict(
        color=stairs.get_edgecolor(), linestyle="none", alpha=0.5, **errorbar_kw
    )
    plt.errorbar(x, h, yerr=dh, **errorbar_kw)
    return h, bins, stairs


def histogram_err(
    a,
    sigma_bounds=(-1, 1),
    bins=None,
    range=None,
    weights=None,
    density=False,
):
    """Return (h, dh, bins) for a histogram of x with error estimates.

    See numpy.histogram for parameters.

    Other Parameters
    ----------------
    sigma_bounds : (float, float)
        Confidence region to plot expressed in terms of 1D normal sigma percentiles.
    """
    unknown_params = set(inspect.signature(np.histogram).parameters).difference(
        {"a", "bins", "range", "weights", "density"}
    )
    if unknown_params:
        warnings.warn(
            "Unknown parameters {unknown_params}: Assumptions about histogram may be invalid"
        )

    assert len(sigma_bounds) == 2
    assert sigma_bounds[0] <= 0
    assert 0 <= sigma_bounds[1]
    percentiles = 100 * sp.stats.norm().cdf(sigma_bounds)

    a = np.asarray(a)
    h, bins = np.histogram(a, bins=bins, range=range, density=density, weights=weights)
    if range is None:
        N = len(a)
    else:
        low, high = range
        N = sum((low < a) & (a <= high))

    # Midpoints and width of the bins
    x = 0.5 * (bins[1:] + bins[:-1])
    dx = np.diff(bins)
    if density:
        p = h * dx
    else:
        p = h / sum(h)
    assert np.allclose(1, sum(p))

    dp = sp.stats.binom(N, p).ppf(percentiles[:, None] / 100) / N - p
    dp[0] *= -1
    assert np.all(0 <= dp)

    if density:
        dh = dp / dx
    else:
        dh = sum(h) * dp

    return h, dh, bins
