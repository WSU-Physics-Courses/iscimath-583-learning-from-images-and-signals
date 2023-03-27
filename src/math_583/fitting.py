"""Tools for model fitting and parameter estimation."""

import numpy as np
import scipy.stats
import scipy.interpolate

sp = scipy

__all__ = ["calibrate"]


def calibrate(
    get_estimation, a, Ns=5000, sigma_levels=(-1, 1), use_mean=False, invert=False
):
    """Return calibration curves for the specified confidence region.

    Arguments
    ---------
    get_estimation : function
        Should return an estimation `A = get_estimation(a)` of the parameter `a`
        inserting appropriate noise.  Thus, typically this will be a composition of
        `A = estimator(get_data(a))` where `A = estimator(y)` provides an estimate of
        `A` from randomly generated data `y = get_data(a)`.
    a : array-like
        Array of parameter values over which to perform the calibration.
    Ns : int
        Number of samples.
    sigma_levels : [float]
        Tuple of sigma levels specifying the percentiles (in terms of the 1D normal
        distribution, so sigma_levels = (-1, 1) will return the 15.9% and 84.1%
        percentiles, expressing a 68.3% confidence interval.
    use_mean : bool
        If True, then use the mean of the samples rather than the median to calibrate
        the central value of the parameter estimate.
    invert : bool
        If False, then return the calibration data (A, c0, c1, ...) evaluated at the
        parameter values a.  If True, then also return the inverse calibration
        ((A, c0, c1, ...), (a_, c0_, c1_, ..._)) as splines where everything is a
        function of the estimator A_. Note: this requires that the estimator and
        confidence levels depend monotonically on parameter.

    Returns
    -------
    (A, c0, c1, c2, ...) : array
        If invert is False, then these are the estimator and confidence levels evaluated
        at the specified parameter values.
    (a_, c0_, c1_, ...) : [InterpolatedUnivariateSpline]
        If invert is True, then these splines are also return (can be called as
        functions) that calculates calibrated parameter estimate a_(A) (mean or median
        of the samples) as a function of the raw estimator A.

    Thus, if `invert=True` and `sigma_levels = (-1, 1)`, then, given an estimate
    `A = A(y)` of some data, we expect that c0_(A) < a < c1_(A) 68.3% of the time.
    """

    if sigma_levels is None or len(sigma_levels) == 0:
        percentiles = [50]
    else:
        percentiles = [50] + (100 * sp.stats.norm().cdf(sigma_levels)).tolist()

    calibration = []
    as_ = [a] if np.isscalar(a) else a
    for _a in as_:
        As = [get_estimation(_a) for n in range(Ns)]
        cl = np.percentile(As, percentiles)
        if use_mean:
            cl[0] = np.mean(As)
        calibration.append(cl)
    calibration = np.transpose(calibration)
    if np.isscalar(a):
        return calibration.ravel()

    if invert:
        Spline = sp.interpolate.InterpolatedUnivariateSpline
        if np.all(np.diff(calibration, axis=1) < 0):
            # Reverse order so calibrations are monotonically increasing.
            a = a[::-1]
            calibration = calibration[:, ::-1]
        if not np.all(np.diff(calibration, axis=1) > 0):
            raise NotImplementedError(
                "Calibration is not Monotonic.  Perhaps try increasing Ns?"
            )
        splines = [Spline(c, a, ext="raise") for c in calibration]
        return calibration, splines
    return calibration
