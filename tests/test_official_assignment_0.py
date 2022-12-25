"""Official Tests for Assignment 0

These tests are the official tests for assignment 0.  Run with:

```bash
pytest -k test_official_assignment_0 --no-cov
```
"""
import numpy as np

import pytest

from math_583 import assignment_0


@pytest.fixture(
    params=[
        (0, 0),
        (1, 1),
        (-1j, 1j),
        (1 - 1j, 1 + 1j),
        (1e-5, 1e10),
    ]
)
def roots(request):
    yield request.param


def test1(roots):
    x1, x2 = roots
    a = 1
    b = -(x1 + x2)
    c = x1 * x2

    xs = assignment_0.quadratic_equation(a=a, b=b, c=c)

    # Allow roots to be in either order
    assert any([np.allclose(roots, xs), np.allclose(roots, xs[::-1])])
