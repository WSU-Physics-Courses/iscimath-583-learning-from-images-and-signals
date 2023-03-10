from itertools import product
import numpy as np

import pytest

from math_583 import denoise


_EPS = np.finfo(float).eps


@pytest.fixture(params=["wrap", "reflect", "periodic"])
def mode(request):
    yield request.param


@pytest.fixture(params=[0.1, 1.0, 2.0])
def lam(request):
    yield request.param


@pytest.fixture(params=[False, True])
def use_shortcuts(request):
    yield request.param


@pytest.fixture(params=[False, True])
def subtract_mean(request):
    yield request.param


class TestDenoise:
    def test_solve_minimize(self, mode, lam):
        im = denoise.Image()
        d = denoise.Denoise(image=im, mode=mode, lam=lam)
        tol = 1e-8
        u_min = d.minimize(tol=tol)
        if mode not in d._K2:
            with pytest.raises(NotImplementedError, match=rf"{mode=} not in"):
                u_solve = d.solve()
            return
        u_solve = d.solve()
        assert np.allclose(u_min, u_solve, atol=4 * np.sqrt(tol))

    def test_derivative(self, mode, use_shortcuts):
        """Test that _df() computes the derivative of _f()."""
        im = denoise.Image()
        for p, q in [(2.0, 2.0), (1.5, 1.5), (1.0, 1.0)]:
            d = denoise.Denoise(
                image=im, mode=mode, p=p, q=q, use_shortcuts=use_shortcuts
            )
            for u in [d.u_exact, d.u_noise]:
                y = d.pack(u)
                dy = im.rng.normal(size=y.shape)
                dy /= np.linalg.norm(dy)
                h = (3 * _EPS) ** (1 / 3)
                d1 = (d._f(y + h * dy) - d._f(y - h * dy)) / 2 / h
                d2 = d._df(y).dot(dy)
                print(mode, (p, q), d1, d2, d1 / d2 - 1)
                assert np.allclose(d1, d2)

    def test_gradient_magnitude(self, mode, use_shortcuts):
        im = denoise.Image()
        d = denoise.Denoise(image=im, mode=mode, use_shortcuts=use_shortcuts)
        for u in [d.u_exact, d.u_noise]:
            du_mag1 = d.gradient_magnitude(u)
            du_mag2 = np.sqrt(d.gradient_magnitude2(u))
            du_mag3 = np.sqrt(np.sum(abs(d.gradient(u, real=False)) ** 2, axis=0))
            assert np.allclose(du_mag1, du_mag2)
            assert np.allclose(du_mag1, du_mag3)

    def test_divergence(self, mode, use_shortcuts):
        im = denoise.Image()
        d = denoise.Denoise(image=im, mode=mode, use_shortcuts=use_shortcuts)
        for u in [d.u_exact, d.u_noise]:
            d2u = d.divergence(d.gradient(u, real=False))
            d2u_ = d.laplacian(u)
            assert np.allclose(d2u, d2u_)


class TestNonLocalMeans:
    def test_pad(self):
        u = np.array([[1, 2, 3], [4, 5, 6]])
        dx = dy = 5
        us_ = {
            "wrap": np.array(
                [
                    [2, 3, 1, 2, 3, 1, 2],
                    [5, 6, 4, 5, 6, 4, 5],
                    [2, 3, 1, 2, 3, 1, 2],
                    [5, 6, 4, 5, 6, 4, 5],
                    [2, 3, 1, 2, 3, 1, 2],
                    [5, 6, 4, 5, 6, 4, 5],
                ]
            ),
            "constant": np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 2, 3, 0, 0],
                    [0, 0, 4, 5, 6, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                ]
            ),
            "reflect": np.array(
                [
                    [5, 4, 4, 5, 6, 6, 5],
                    [2, 1, 1, 2, 3, 3, 2],
                    [2, 1, 1, 2, 3, 3, 2],
                    [5, 4, 4, 5, 6, 6, 5],
                    [5, 4, 4, 5, 6, 6, 5],
                    [2, 1, 1, 2, 3, 3, 2],
                ]
            ),
            "nearest": np.array(
                [
                    [1, 1, 1, 2, 3, 3, 3],
                    [1, 1, 1, 2, 3, 3, 3],
                    [1, 1, 1, 2, 3, 3, 3],
                    [4, 4, 4, 5, 6, 6, 6],
                    [4, 4, 4, 5, 6, 6, 6],
                    [4, 4, 4, 5, 6, 6, 6],
                ],
            ),
            "mirror": np.array(
                [
                    [3, 2, 1, 2, 3, 2, 1],
                    [6, 5, 4, 5, 6, 5, 4],
                    [3, 2, 1, 2, 3, 2, 1],
                    [6, 5, 4, 5, 6, 5, 4],
                    [3, 2, 1, 2, 3, 2, 1],
                    [6, 5, 4, 5, 6, 5, 4],
                ]
            ),
        }
        us_["periodic"] = us_["wrap"]

        for mode in us_:
            nlm = denoise.NonLocalMeans(denoise.Image(u), mode=mode)
            assert np.allclose(nlm.pad(u), us_[mode])

    def test_get_threshold(self, subtract_mean):
        sigma = 0.4
        for dx, dy, percentile in product(range(2, 5), range(2, 5), [50, 98]):
            nlm = denoise.NonLocalMeans(
                denoise.Image(), sigma=sigma, dx=dx, dy=dy, subtract_mean=subtract_mean
            )
            Nsamples = 1000
            th_ = nlm.get_threshold(percentile=percentile)
            okay = False
            for n in range(4):
                # Try 4 times.
                th = nlm.get_threshold(percentile=percentile, Nsamples=Nsamples)
                okay = np.allclose(th_, th, rtol=2 / np.sqrt(Nsamples))
                if okay:
                    break
            assert np.allclose(th_, th, rtol=2 / np.sqrt(Nsamples))
