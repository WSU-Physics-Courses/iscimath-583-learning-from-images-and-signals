"""Module with tools for exploring image denoising.
"""
import logging
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
import scipy.optimize

import mmf_setup

import PIL

logging.getLogger("PIL").setLevel(logging.ERROR)  # Suppress PIL messages

sp = scipy

plt.rcParams["image.cmap"] = "gray"  # Use greyscale as a default.


class Base:
    """Base class for setting attributes."""

    def __init__(self, **kw):
        for key in kw:
            if not hasattr(self, key):
                raise ValueError(f"Unknown {key=}")
            setattr(self, key, kw[key])
        self.init()

    def init(self):
        return


class Image(Base):
    """Class to load and process images."""

    dir = Path(mmf_setup.ROOT) / ".." / "_data" / "images"
    filename = "The-original-cameraman-image.png"
    seed = 2

    def init(self):
        self.rng = np.random.default_rng(seed=2)
        self._filename = Path(self.dir) / self.filename
        self.image = PIL.Image.open(self._filename)
        self.shape = self.image.size[::-1]

    @property
    def rgb(self):
        """Return the RGB form of the image."""
        return np.asarray(self.image.convert("RGB"))

    def get_data(self, normalize=False, sigma=0, rng=None):
        """Return greyscale image.

        Arguments
        ---------
        normalize : bool
            If `True`, then normalize the data so it is between 0 and 1.
        sigma : float
            Standard deviation (as a fraction) of gaussian noise to add to the image.
            The result will be clipped so it does not exceed (0, 255) or (0, 1) if
            `normalize==True`.
        """
        data = np.asarray(self.image.convert("L"))
        vmin, vmax = 0, 255
        if normalize:
            data = data / vmax
            vmax = 1.0

        if sigma:
            if rng is None:
                rng = self.rng
            eta = sigma * rng.normal(size=data.shape)
            if normalize:
                data += eta
            else:
                data = vmax * eta + data
            data = np.minimum(np.maximum(data, vmin), vmax)
            if not normalize:
                data = data.round(0).astype("uint8")
        return data

    @property
    def data(self):
        """Return a greyscale image with data between 0 and 255."""
        return np.asarray(self.image.convert("L"))

    def __repr__(self):
        return self.image.__repr__()

    def _repr_pretty_(self, *v, **kw):
        return self.image._repr_pretty_(*v, **kw)

    def _repr_png_(self):
        """Use the image as the representation for IPython display purposes."""
        return self.image._repr_png_()

    def imshow(self, u, vmin=None, vmax=None, ax=None, **kw):
        if vmax is None:
            if u.dtype == np.dtype("uint8"):
                vmax = max(255, u.max())
            else:
                vmax = max(1.0, u.max())
        if vmin is None:
            vmin = min(0, u.min())

        if ax is None:
            ax = plt.gca()

        ax.imshow(u, vmin=vmin, vmax=vmax, **kw)
        ax.axis("off")


class Denoise(Base):
    lam = 1.0
    mode = "reflect"
    image = None
    sigma = 0.5

    def init(self):
        self.u_exact = self.image.get_data(sigma=0)
        self.u_noise = self.image.get_data(sigma=self.sigma)

    def laplacian(self, u):
        """Return the laplacian of u."""
        return sp.ndimage.laplace(u, mode=self.mode)

    def _u_noise(self, u_noise):
        if u_noise is None:
            u_noise = self.u_noise
        return np.asarray(u_noise)

    def get_energy(self, u, u_noise=None):
        """Return the energy."""
        u_noise = self._u_noise(u_noise)
        E_regularization = (-u * self.laplacian(u)).sum()
        E_data_fidelity = (np.abs(u - u_noise) ** 2).sum()
        print(f"{E_regularization=}, {E_data_fidelity=}")
        return E_regularization + self.lam * E_data_fidelity

    def pack(self, u):
        """Return y, the 1d real representation of u for solving."""
        return np.ravel(u)

    def unpack(self, y, u_noise=None):
        """Return `u` from the 1d real representation y."""
        u_noise = self._u_noise(u_noise)
        return np.reshape(y, u_noise.shape)

    def _f(self, y):
        """Return the energy"""
        return self.get_energy(self.unpack(y))

    def _df(self, y, u_noise=None):
        """Return the gradient of f(y)."""
        u = self.unpack(y)
        u_noise = self._u_noise(u_noise)
        return self.pack(2 * (-self.laplacian(u) + self.lam * (u - u_noise)))

    def compute_dy_dt(self, t, y):
        """Return dy_dt for the solver."""
        return -self.beta * self.df(y=y)

    def callback(self, y, state):
        u = self.unpack(y)

    def minimize(self, u0=None, method="BFGS", **kw):
        y0 = self.pack(self._u_noise(u0))
        res = sp.optimize.minimize(self.f, x0=y0, jac=self.df, method=method, **kw)
        return res
