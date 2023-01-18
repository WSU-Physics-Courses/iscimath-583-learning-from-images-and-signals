"""Tools for exploring image denoising.
"""
from functools import partial
import logging
import os.path
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
import scipy.optimize

import PIL

logging.getLogger("PIL").setLevel(logging.ERROR)  # Suppress PIL messages

sp = scipy

plt.rcParams["image.cmap"] = "gray"  # Use greyscale as a default.


__all__ = ["subplots", "Image", "Denoise"]


def subplots(cols=1, rows=1, height=3, aspect=1, **kw):
    """More convenient subplots that automatically sets the figsize.

    Arguments
    ---------
    cols, rows : int
        Number of columns and rows in the figure.
    height : float
        Height of an individual element.  Unless specified, the figure size will be
       ``figsize=(cols * height * aspect, rows * height)``.
    aspect : float
        Aspect ratio of each figure (width/height).
    **kw : dict
        Other arguments are passed to ``plt.subplot()``
    """
    args = dict(figsize=(cols * height, rows * height))
    args.update(kw)
    return plt.subplots(rows, cols, **args)


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

    if os.path.exists("images"):
        # Use a local directory if it exists.  Does not need mmf_setup
        dir = Path("images")
    else:
        # Otherwise (i.e. for documentation) go relative to ROOT
        import mmf_setup

        mmf_setup.set_path()
        dir = Path(mmf_setup.ROOT) / ".." / "_data" / "images"

    filename = "The-original-cameraman-image.png"
    seed = 2

    def init(self):
        self.rng = np.random.default_rng(seed=self.seed)
        self._filename = Path(self.dir) / self.filename
        self.image = PIL.Image.open(self._filename)
        self.shape = self.image.size[::-1]

    @property
    def rgb(self):
        """Return the RGB form of the image."""
        return np.asarray(self.image.convert("RGB"))

    def get_data(self, normalize=True, sigma=0, rng=None):
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

    def show(self, u, vmin=None, vmax=None, ax=None, **kw):
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

    imshow = show


class Denoise(Base):
    """Class for denoising images.

    Attributes
    ----------
    image : Image
        Instance of :class:`Image` with the image data.
    lam : float
        Parameter λ controlling the regularization.  Larger values will produce an image
        closer to the target.  Smaller values will produce smoother images.
    """

    lam = 1.0
    mode = "reflect"
    image = None
    sigma = 0.5
    seed = 2

    def init(self):
        self.rng = np.random.default_rng(seed=self.seed)
        self.u_exact = self.image.get_data(sigma=0, normalize=True)
        self.u_noise = self.image.get_data(
            sigma=self.sigma, normalize=True, rng=self.rng
        )

        # Compute Fourier momenta
        dx = 1.0
        self._kxyz = np.meshgrid(
            *[2 * np.pi * np.fft.fftfreq(_N, dx) for _N in self.image.shape],
            sparse=True,
            indexing="ij",
        )

        self._K2 = {
            "periodic": sum(_k**2 for _k in self._kxyz),
            "wrap": 2 * sum(1 - np.cos(_k * dx) for _k in self._kxyz) / dx**2,
        }

        # Pre-compute some energies for normalization.
        self._E_noise = self.get_energy(self.u_noise, parts=True, normalize=False)
        self._E_exact = self.get_energy(self.u_exact, parts=True)

    def _fft(self, u):
        # Could replace with an optimize version for pyfftw.
        return np.fft.fftn(u)

    def _ifft(self, u):
        # Could replace with an optimize version for pyfftw.
        return np.fft.ifftn(u)

    def laplacian(self, u):
        """Return the laplacian of u."""
        if self.mode == "periodic":
            res = self._ifft(-self._K2[self.mode] * self._fft(u))
            assert np.allclose(0, res.imag)
            return res.real
        return sp.ndimage.laplace(u, mode=self.mode)

    def get_energy(self, u, parts=False, normalize=True):
        """Return the energy.

        Arguments
        ---------
        parts : bool
            If True, return (E, E_regularization, E_data_fidelity)
        normalize : bool
            If True, normalize by the starting values for u_noise.
        """
        u_noise = self.u_noise
        E_regularization = (-u * self.laplacian(u)).sum() / 2
        E_data_fidelity = (abs(u - u_noise) ** 2).sum() / 2
        E = E_regularization + self.lam * E_data_fidelity
        E0 = self.lam * np.prod(u_noise.shape)
        if normalize:
            E0 = self._E_noise[0]

        if parts:
            return (E / E0, E_regularization / E0, E_data_fidelity / E0)
        else:
            return E / E0

    def pack(self, u):
        """Return y, the 1d real representation of u for solving."""
        return np.ravel(u)

    def unpack(self, y):
        """Return `u` from the 1d real representation y."""
        return np.reshape(y, self.u_noise.shape)

    def compute_dy_dt(self, t, y):
        """Return dy_dt for the solver."""
        return -self.beta * self._df(y=y)

    def _f(self, y):
        """Return the energy"""
        return self.get_energy(self.unpack(y), normalize=True)

    def _df(self, y):
        """Return the gradient of f(y)."""
        E0 = self._E_noise[0]
        u = self.unpack(y)
        return self.pack(-self.laplacian(u) + self.lam * (u - self.u_noise)) / E0

    def callback(self, y, plot=False):
        u = self.unpack(y)
        E, E_r, E_f = self.get_energy(u, parts=True)

        msg = f"E={E:.2g}, E_r={E_r:.2g}, E_f={E_f:.2g}"
        if plot:
            import IPython.display

            fig = plt.gcf()
            ax = plt.gca()
            ax.cla()
            IPython.display.clear_output(wait=True)
            self.image.show(u, ax=ax)
            ax.set(title=msg)
            IPython.display.display(fig)
        else:
            print(msg)

    def minimize(
        self, u0=None, method="L-BFGS-B", callback=True, tol=1e-8, plot=False, **kw
    ):
        """Directly solve the minimization problem with the L-BFGS-B method."""
        if u0 is None:
            u0 = self.u_noise
        y0 = self.pack(u0)
        if callback:
            callback = partial(self.callback, plot=plot)
        res = sp.optimize.minimize(
            self._f,
            x0=y0,
            jac=self._df,
            method=method,
            callback=callback,
            tol=tol,
            **kw,
        )
        if not res.success:
            raise Exception(res.message)

        if plot:
            plt.close("all")
        u = self.unpack(res.x)
        return u

    def solve(self):
        """Directly solve the minimization problem using Fourier techniques.

        This is much faster than running `minimize` but only supports limited modes.
        """
        mode = self.mode
        if mode not in self._K2:
            raise NotImplementedError(f"{mode=} not in {set(self._K2)}")
        res = self._ifft(self._fft(self.u_noise) / (self._K2[mode] / self.lam + 1))
        assert np.allclose(res.imag, 0)
        return res.real
