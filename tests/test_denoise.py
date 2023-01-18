import numpy as np

import pytest

from math_583 import denoise


@pytest.fixture(params=["wrap", "reflect", "periodic"])
def mode(request):
    yield request.param


@pytest.fixture(params=[0.1, 1.0, 2.0])
def lam(request):
    yield request.param


class TestDenoise:
    def test_derivative(self, mode):
        """Test that _df() computes the derivative of _f()."""
        im = denoise.Image()
        d = denoise.Denoise(image=im, mode=mode)
        for u in [d.u_exact, d.u_noise]:
            y = d.pack(u)
            dy = im.rng.normal(size=y.shape)
            dy /= np.linalg.norm(dy)
            h = 0.0001
            assert np.allclose(
                (d._f(y + h * dy) - d._f(y - h * dy)) / 2 / h, d._df(y).dot(dy)
            )

    def test_solve_minimize(self, mode, lam):
        im = denoise.Image()
        d = denoise.Denoise(image=im, mode=mode, lam=lam)
        tol = 1e-8
        u_min = d.minimize(tol=tol)
        if mode not in d._K2:
            with pytest.raises(NotImplementedError, match=rf"{mode=} not in") as e:
                u_solve = d.solve()
            return
        u_solve = d.solve()
        assert np.allclose(u_min, u_solve, atol=4 * np.sqrt(tol))
