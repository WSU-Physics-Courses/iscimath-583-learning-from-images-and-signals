"""Assignment 0

Evaluation of special functions.
"""
import cmath
import numpy as np

_TINY = np.finfo(float).tiny


@np.vectorize
def quadratic_equation(a, b, c):
    """Return `xs=(x1, x2)`, the two roots of the quadratic equation $ax^2+bx*c=0$.

    Examples
    --------
    >>> quadratic_equation(1, 2, 3)

    """
    d = b ** 2 - 4 * a * c
    x1 = (-b - cmath.sqrt(d)) / 2 / a
    x2 = (-b + cmath.sqrt(d)) / 2 / a
    return (x1, x2)


def _quadratic_equation(a, b, c):  # pragma: no cover
    """Return `xs=(x1, x2)`, the two roots of the quadratic equation $ax^2+bx*c=0$.

    This version is stable with respect to cancellation errors.
    """
    sqd = np.sqrt(b ** 2 - 4 * a * c + 0j)
    m, p = -b - sqd, -b + sqd
    x1 = np.where(abs(m) > abs(p), m, p) / 2 / a
    x2 = c / (x1 + _TINY)
    return (x1, x2)
