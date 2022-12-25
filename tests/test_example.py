import numpy as np

# Sets the path based on the config file.
import mmf_setup

mmf_setup.set_path()

from math_583 import example

import pytest


class TestDifferentiate:
    def test_real(self):
        x = np.linspace(-10, 10, 100)
        dx = np.diff(x).mean()
        f = np.exp(-(x ** 2) / 2)
        df_exact = -x * f
        df = example.differentiate_fft(f, dx)
        assert np.allclose(df, df_exact)

    def test_complex(self):
        x = np.linspace(-10, 10, 100)
        dx = np.diff(x).mean()
        f = np.exp(-(x ** 2) / 2) * (1.1 + 1.2j * x)
        df_exact = -x * f + 1.2j * np.exp(-(x ** 2) / 2)
        df = example.differentiate_fft(f, dx)
        assert np.allclose(df, df_exact)
