"""Basic Tests for Assignment 0.
"""
import numpy as np

# import pytest

from math_583 import assignment_0


def test_arrays():
    """Make sure code works with arrays."""
    np.random.seed(3)
    a, b, c = np.random.random((3, 4)) - 0.5
    x1, x2 = assignment_0.quadratic_equation(a=a, b=b, c=c)
    for x in (x1, x2):
        res = x * (a * x + b) + c
        assert np.allclose(res, 0)
