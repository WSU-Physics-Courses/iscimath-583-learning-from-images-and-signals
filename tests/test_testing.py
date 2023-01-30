import numpy as np

import pytest

from math_583 import testing

_EPS = np.finfo(float).eps


@pytest.fixture(params=[0, 0.5, 1.0])
def eta(request):
    yield request.param


class TestFunctions:
    def test_f_eta(self, eta):
        L = 1.23
        f = testing.Functions(L=L, eta=eta)
        x = np.linspace(0, L, 1000)
        assert np.allclose(f(x), f(x + L))

        f_x = f(x)
        df_x = f(x, d=1)
        ddf_x = f(x, d=2)

        h = (3 * _EPS) ** (1 / 3)
        df_ = (f(x + h) - f(x - h)) / 2 / h
        print(abs(df_ - df_x).max())
        assert np.allclose(df_, df_x)

        ddf_ = (f(x + h, d=1) - f(x - h, d=1)) / 2 / h
        print(abs(ddf_ - ddf_x).max())
        assert np.allclose(ddf_, ddf_x)
