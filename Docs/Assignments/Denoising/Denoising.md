---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.4
kernelspec:
  display_name: Python 3 (math-583)
  language: python
  name: math-583
---

```{code-cell} ipython3
:tags: [hide-cell]

%matplotlib inline
import numpy as np, matplotlib.pyplot as plt
import mmf_setup; mmf_setup.nbinit()
import logging
logging.getLogger('PIL').setLevel(logging.ERROR)  # Suppress PIL messages
import PIL
```

# Denoising

## The Problem

## Walkthrough

### Loading Images

```{code-cell} ipython3
# See what images are available

!ls images/
```

Use the [Python Imaging Library](https://python-pillow.org/) (PIL) to load images:

```{code-cell} ipython3
# Use the PIL to load the image
from PIL import Image

im = Image.open("images/The-original-cameraman-image.png")
im  # or display(im), but this happens by default for the last line
```

### Adding Noise

```{code-cell} ipython3
# First, see what the data looks like as an array.
# IMPORTANT: Convert to "L", "RGB" etc. or things get wierd:-)
# We also normalize pixel values in [0, 1)
u = u_exact = np.asarray(im.convert("L")) / 256
display(u)
display((u.shape, u.max(), u.min()))
print(f"{u.shape=}, {u.max()=}, {u.min()=}")  # Nicer python f-string
```

We see that the image is 490×487 in shape, and has unsigned integers ranging from 0 to 255.

```{code-cell} ipython3
# We can also display the array
plt.imshow(u, vmin=0, vmax=1, cmap="gray", interpolation=None)
```

```{code-cell} ipython3
# Use random numbers to add noise.  Here we get a random number generator (rng)
# and use a fixed seed so we can reproduce our results
rng = np.random.default_rng(seed=2)
sigma = 0.3
u_noise = u + sigma * rng.normal(size=u.shape)
plt.imshow(u_noise, vmin=0, vmax=1, cmap="gray")
```

### Removing Noise

```{code-cell} ipython3
def imshow(u):
    fig, ax = plt.subplots()
    im = ax.imshow(u, vmin=0, vmax=1, cmap='gray')
    return im
```

```{code-cell} ipython3
def laplacian(u):
    """Return the laplacian of the image usings unit spacing."""
    return sum(
        np.gradient(
            np.gradient(u, axis=_a, edge_order=2), axis=_a, edge_order=2)
        for _a in [0, 1])
    return (np.gradient(np.gradient(u, axis=0), axis=0) +
            np.gradient(np.gradient(u, axis=1), axis=1))

import scipy.ndimage

def laplacian(u):
    return scipy.ndimage.laplace(u)
```

```{code-cell} ipython3
from ipywidgets import interact

d2u = laplacian(u_noise)

#@interact(p=(0, 50, 1))
def go(p=10):
    vmin, vmax = np.percentile(d2u, [p, 100-p])
    plt.imshow(d2u, vmin=vmin, vmax=vmax, cmap='gray')
```

We will minimize

\begin{gather}
  E[u] = \int\abs{\vect{\nabla} u}^2 + \lambda \int \abs{u - d}^2,\\
  dE(u) = \pdiff{E}{u} = 2\Bigl(-\nabla^2u + \lambda (u-d)\Bigr) = 0,
\end{gather}
where $d\equiv$`u_noise` is the **data**, or the noisy image. We can do this with a
direct gradient descent:

\begin{gather}
  \diff{u}{t} = -\beta\; dE(u)
\end{gather}

```{code-cell} ipython3
from scipy.integrate import solve_ivp

beta = 1.0
lam = 1.0  # Lam

def compute_E(u, lam=lam):
    """Return the energy."""
    E_regularization = (-u * laplacian(u)).sum()
    E_data_fidelity = (np.abs(u-u_noise)**2).sum()
    print(f"{E_regularization=}, {E_data_fidelity=}")
    return E_regularization + lam * E_data_fidelity

def compute_dy_dt(t, y, beta=beta, lam=lam):
    """Return dy_dt."""
    # solve_ivp expects y to be 1D, so we need to reshape
    u = y.reshape(u_noise.shape)
    du = 2*(-laplacian(u) + lam * (u - u_noise))
    dy = np.ravel(- beta * du)
    return dy


# Make initial guess 1D.  We start with the noise.
y0 = np.ravel(u_noise)
dT = 10
res = solve_ivp(compute_dy_dt, y0=y0, t_span=(0, dT))
u1 = res.y[:, -1].reshape(u_noise.shape)
plt.imshow(u1, vmin=0, vmax=1, cmap='gray')
```

```{code-cell} ipython3
from functools import partial
from IPython.display import clear_output
import tqdm

dT = 1
lam = 0.001  # Lam
lam = 1/2.2  # Lam
steps = 10

err0 = (np.abs(u_noise - u_exact)**2).mean()
E0 = compute_E(u_noise, lam=lam)

errs = []
Es = []
ts = []
y = y0
t = 0
fig, ax = plt.subplots()
for step in tqdm.tqdm(range(steps)):
    res = solve_ivp(partial(compute_dy_dt, beta=beta, lam=lam), y0=y, t_span=(t, t+dT))
    y = res.y[:, -1]
    u = y.reshape(u_noise.shape)
    errs.append((np.abs(u - u_exact)**2).mean()/err0)
    Es.append(compute_E(u, lam=lam)/E0)
    t = res.t[-1]
    ts.append(t)
    ax.cla()
    ax.imshow(u, vmin=0, vmax=1, cmap='gray')
    clear_output(wait=True)
    display(fig)
plt.close('all')
```

```{code-cell} ipython3
fig, ax = plt.subplots()
ax.plot(ts, errs, label='err/err0')
ax.plot(ts, Es, label='E/E0')
ax.legend()
ax.set(xlabel=r"$\beta t$");  # Python r-string
```

```{code-cell} ipython3
fig, axs = plt.subplots(1, 3, figsize=(10, 5))
ax = axs[0]
ax.imshow(u_noise, vmin=0, vmax=1, cmap="gray")
ax = axs[1]
ax.imshow(u, vmin=0, vmax=1, cmap="gray")
ax = axs[2]
ax.imshow(u_exact, vmin=0, vmax=1, cmap="gray")
```

## Exploration

Here we explore some properties of denoising using the tools in `denoise.py`, which
packages code similar to that use above into classes and functions for easy reuse,
testing, and exploration.

We start with an image and various levels of noise:

```{code-cell} ipython3
%matplotlib inline
import mmf_setup;mmf_setup.set_path()
import numpy as np, matplotlib.pyplot as plt
import PIL
from importlib import reload
import denoise;reload(denoise)

im = denoise.Image()
sigmas = np.linspace(0, 1, 6)
fig, axs = plt.subplots(1, len(sigmas), figsize=(len(sigmas)*2, 2))
for sigma, ax in zip(sigmas, axs):
    plt.sca(ax)
    im.imshow(im.get_data(sigma=sigma))
    ax.set(title=f"{sigma=:.2g}")
```

```{code-cell} ipython3
import denoise;reload(denoise)
im = denoise.Image()
d = denoise.Denoise(image=im, sigma=0.4, lam=1000.0)
u = d.u_exact
y = d.pack(u)

dy = im.rng.normal(size=y.shape)
dy /= np.linalg.norm(dy)
```

```{code-cell} ipython3
h = 0.0001
(d._f(y+h*dy) - d._f(y-h*dy))/2/h, d._df(y).dot(dy)
#d._f(y), d._df(y)
```

## Source Code

The following cells contain the source code for modules used in this document.  They
need to be executed once to generate the source files, but afterwards can just be
imported.  They use the [`%%file`]() magic which writes the contents to file.

```{code-cell} ipython3
#%%writefile denoise.py
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import PIL


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
    dir = "images"
    filename = "The-original-cameraman-image.png"

    def init(self):
        self._filename = Path(self.dir) / self.filename
        self.image = PIL.Image.open(self._filename)

    @property
    def rgb(self):
        """Return the RGB form of the image."""
        return self.image.convert("RGB")

    @property
    def data(self):
        """Return a greyscale image with data between 0 and 255."""
        return self.image.convert("L")
```

```{code-cell} ipython3
%load_ext autoreload
%autoreload
import denoise

im = denoise.Image()
```

```{code-cell} ipython3
import scipy.ndimage
u = im.get_data(sigma=0)
%timeit d2u1 = scipy.ndimage.laplace(u, mode='reflect')
%timeit d2u2 = scipy.ndimage.laplace(u, mode='wrap')
#plt.imshow(abs(d2u1-d2u2))
#plt.colorbar()
%timeit np.fft.ifftn(np.fft.fft2(u))
```

```{code-cell} ipython3
scipy.ndimage.gaussian_laplace??
```

```{code-cell} ipython3

```
