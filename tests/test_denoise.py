import numpy as np

import pytest

from math_583 import denoise


class TestDenoise:

    def test_derivative(self):
        """Test that _df() computes the derivative of _f()."""
        im = denoise.Image()
        d = denoise.Denoise(image=im)
        for u in [d.u_exact, d.u_noise]:
            y = d.pack(u)
            dy = im.rng.normal(size=y.shape)
            dy /= np.linalg.norm(dy)
            h = 0.0001
            assert np.allclose((d._f(y + h * dy) - d._f(y - h * dy)) / 2 / h,
                               d._df(y).dot(dy))
