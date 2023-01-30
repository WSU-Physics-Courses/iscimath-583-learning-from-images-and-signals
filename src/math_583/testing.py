"""Tools for testing."""
import numpy as np


class Functions:
    """Various functions like f_η(x) used for testing in the documentation.

    This function is periodic with period `L`, :math:`C^∞`, analytic for `eta<1` and
    non-analytic for eta=1.

    Attributes
    ----------
    L : float
        Period.
    eta : float
        Smoothness parameter.  eta=0 is a constant function, 0<eta<1 is analytic.  eta=1
        is :math:`C^∞` but non-analytic at the middle of the interval.
    """

    def __init__(self, L=1.0, eta=0.5):
        self.L = L
        self.eta = eta
        self.k = 2 * np.pi / L

    def f(self, x, d=0):
        """Return the d'th derivative."""
        eta, k = self.eta, self.k
        c = np.cos(k * x)
        denom = 1 + eta * c
        f = np.exp(-1 / denom)
        if d == 0:
            return f
        s = np.sin(k * x)
        if d == 1:
            return -eta * k * s / denom**2 * f
        if d == 2:
            return (
                -eta
                * k**2
                * (eta * (1 + c**2 - eta * c**3) + (1 + 2 * eta**2) * c)
                / denom**4
                * f
            )
        raise NotImplementedError(f"{d=}")

    def __call__(self, x, d=0):
        return self.f(x, d=d)
