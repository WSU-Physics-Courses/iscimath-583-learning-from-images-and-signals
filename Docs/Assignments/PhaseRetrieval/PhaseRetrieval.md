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

```{code-cell}
:tags: [hide-cell]

import mmf_setup; mmf_setup.nbinit()
import logging; logging.getLogger("matplotlib").setLevel(logging.CRITICAL)
%matplotlib inline
import numpy as np, matplotlib.pyplot as plt
```

# Phase Retrieval

Here we consider the following problem of retrieving the phase $\phi(x)$ from a set of images $I_n(x)$ with the model
\begin{gather*}
  I_n(x) = b(x) + a(x)\cos\Bigl(\phi(x) + \theta_n\Bigr) + \eta_n(x),
\end{gather*}
where we assume that $\eta_n(x)$ is noise and relatively small.  We consider two cases:

1. We can control $\theta_n$.
2. $\theta_n$ takes on random unknown values.

+++

## Challenge

+++

As a challenge, consider the following problem with $-10 < x < 10$:

$$
  a(x) = \frac{10}{x^2 + 2}, \qquad
  b(x) = 2\sin(x),\qquad
  \phi(x) = 2x - 10e^{-x^2}.
$$

+++

To simplify the problem, start by assuming no noise $\eta_n = 0$, and that choose $\theta_n$ freely as you like.  For example, if you choose
$$
  \theta_n = \left.\frac{2\pi n}{N}\right|_{n=0}^{N-1}
$$
then averaging $I_n$ over $n$ gives $b(x)$.

Don't worry
about reconstructing the full phase $\phi$: just find $\phi \in [-\pi/2, \pi/2)$ module
$2\pi$.  To reconstruct the smooth $\phi(x)$, one can use the
[`skimage.restoration.unwrap_phase`](https://scikit-image.org/docs/dev/api/skimage.restoration.html#skimage.restoration.unwrap_phase).
(This last step is a little complicated to implement.)

As encouragement, a simple solution with these conditions can be implemented in less
than 20 lines, which works reasonably well even with moderate noise $\eta_n$.

```{code-cell}
%matplotlib inline
import numpy as np, matplotlib.pyplot as plt

x = np.linspace(-10, 10, 256)
b = 2*np.sin(x)
a = 10 / (x**2 + 2)
phi = x/2 - 5 * np.exp(-x ** 2)

fig, axs = plt.subplots(2, 1, sharex=True, figsize=(5, 5))
ax = axs[0]
ax.plot(x, a, "-", label="a")
ax.plot(x, b, "--", label="b")
ax.plot(x, phi, ":", label="φ")
ax.plot(x, np.angle(np.exp(1j*phi)), "-.", label="φ mod 2π")
ax.set(xlabel="x")
ax.legend()


def get_I(theta, a=a, b=b, phi=phi, eta=0, rng=np.random.default_rng(seed=2)):
    if not np.isscalar(theta):
        theta = np.asarray(theta)[:, np.newaxis]
    I = b + a * np.cos(phi + theta)
    I += eta * rng.normal(size=I.shape)
    return I.T  # Transpose so that columns are the images
    
thetas = np.linspace(0, 6 * np.pi, 100)

ax = axs[1]
I = get_I(thetas, eta=1)
ax.pcolormesh(x, thetas, I.T, shading="auto")
ax.set(ylabel=r"θ", xlabel="x");
```

```{code-cell}
# Try to find a solution!

plt.plot(x, get_residual(phi+1))
```

# A Simple Solution

```{code-cell}
from skimage.restoration import unwrap_phase

Np = 10  # Number of images
eta = 0.02  # Noise

# Equally spaced thetas allow us to average to get b(x)
thetas = 2 * np.pi * np.arange(Np) / Np  # Note: np.arange(Np) has 0, 1, ..., Np-1

# Random Number Generator (rng) with fixed seed for reproducible results
rng = np.random.default_rng(seed=2)

# Get the images
I = get_I(thetas, eta=eta, rng=rng)
Nx = len(x)
assert I.shape == (Nx, Np)

# This is not needed, but we check anyway
# Compute the SVD and check that there are only 3 dominant components.
U, λ, Vt = np.linalg.svd(I, full_matrices=False)
if eta < 1e-8:
    assert max(λ[3:]) / min(λ[:3]) < 1e-8
print(f"{λ[3]/λ[2]=:.2g}") # Once this ratio gets too large, we probably can't succeed.


#u = U[:, :3] * λ[np.newaxis, :3]
#vt = Vt[:3, :]

b = I.mean(axis=1)  # Only works for equally spaced points?
Ib = I - b[:, np.newaxis]  # Subtract mean
Ub, λb, Vtb = np.linalg.svd(Ib, full_matrices=False)
if eta < 1e-8:
    assert max(λb[2:]) / min(λb[:2]) < 1e-8
print(f"{λb[2]/λb[1]=:.2g}") # Once this ratio gets too large, we probably can't succeed.
ud = Ub[:, :2] * λb[np.newaxis, :2]
a = np.sqrt((ud**2).sum(axis=1))
cs = ud / a[:, np.newaxis]
phase = np.arctan2(*ud.T)   # Assume we know phi(0) = 0
phase = unwrap_phase(phase)
phase += phi[0] - phase[0]  # Assume we know this!
plt.plot(x, phase-phi)
plt.plot(x, phi)
plt.plot(x, phase)
```

# Exploration

+++

We now re-code the same problem, but a little more modular so that we can explore the sensitivity to errors etc.

```{code-cell}
class Base:
    """Base class to set attributes."""

    def __init__(self, **kw):
        for key in kw:
            if not hasattr(self, key):
                raise ValueError(f"Unknown {key=}")
            setattr(self, key, kw[key])
        self.init()

    def init(self):
        """Overload to compute stuff on initialization."""


class Problem(Base):
    x = np.linspace(-10, 10, 256)
    b = 2 * np.sin(x)
    a = 10 / (x**2 + 2)
    phi = x / 2 - 5 * np.exp(-(x**2))
    rng = np.random.default_rng(seed=2)

    def get_I(self, theta, eta=0, rng=None):
        if rng is None:
            rng = self.rng
        if not np.isscalar(theta):
            theta = np.asarray(theta)[:, np.newaxis]
        I = self.b + self.a * np.cos(self.phi + theta)
        I += eta * rng.normal(size=I.shape)
        return I.T  # Transpose so that columns are the images


class RetrievePhase(Base):
    Np = 10
    
    def get_phase(self, problem, eta=0, rng=None):
        """Return (phase, λb[2]/λb[1])"""
        thetas = 2 * np.pi * np.arange(self.Np) / self.Np
        I = problem.get_I(thetas, eta=eta, rng=rng)
        Nx = len(problem.x)
        assert I.shape == (Nx, self.Np)
        b = I.mean(axis=1)
        Ib = I - b[:, np.newaxis]
        Ub, λb, Vtb = np.linalg.svd(Ib, full_matrices=False)
        ud = Ub[:, :2] * λb[np.newaxis, :2]
        a = np.sqrt((ud**2).sum(axis=1))
        cs = ud / a[:, np.newaxis]
        phase = np.arctan2(*ud.T)  # Assume we know phi(0) = 0
        phase = unwrap_phase(phase)
        
        phase += problem.phi[0] - phase[0]  # Assume we know this!
        return phase, λb[2]/λb[1]

p = Problem()
```

First we explore the sensitivity to noise:

```{code-cell}
r = RetrievePhase(Np=10)

etas = 10**np.linspace(-12, -1, 500)

phases_and_ratios = [r.get_phase(problem=p, eta=_eta) for _eta in etas]

# A little trick from python: phases is a list of (phase, ratio) pairs.
# The python zip function allows use to reorganize this as a list of phases
# and a list ratios.  We then apply np.asarray to make them into arrays
phases, ratios = map(np.asarray, zip(*phases_and_ratios))

fig, ax = plt.subplots(figsize=(6,3))
ax.loglog(etas, abs(phases - p.phi).max(axis=1))
ax.set(xlabel="η", ylabel="Max abs. err. in φ")
ax1 = ax.twinx()
ax1.loglog(etas, ratios, 'C1')
ax1.set(ylabel="$λ_2/λ_1$");
```

We see that the absolute error in the phase retrieval depends linearly on the size of the error.  (Formally, we need to check the exponent, but it is 1).  Second, we see that the ratio $\lambda_2/\lambda_1$ provides a good estimate of the error, with $\lambda_2 < 0.04\lambda_1$ for about percent level accuracy.

These values are probably quite specific to this problem.  With work, one can probably understand exactly how these errors behave.

+++

Now, for a fixed level of noise, can we improve the results by taking more images?

```{code-cell}
2**np.arange(2, 7)
```

```{code-cell}
import tqdm
Nps = np.array([3] + (2**np.arange(2, 7)).tolist())
Nsamples = 100
eta = 1e-4
errs = []
stds = []
for Np in tqdm.tqdm(Nps):
    r = RetrievePhase(Np=Np)
    errs_ = [abs(r.get_phase(problem=p, eta=eta)[0] -  p.phi).max()
             for _sample in range(Nsamples)]
    errs.append(np.mean(errs_))
    stds.append(np.std(errs_))
    
```

```{code-cell}
errs, stds = map(np.asarray, (errs, stds))
fig, ax = plt.subplots(figsize=(6,3))
ax.loglog(1/Nps, errs)
ax.errorbar(1/Nps, errs, stds)
ax.set(xlabel="$N_p$", ylabel="Average max abs. err. in φ");
ax.set_xticks(1/Nps)
ax.set_xticklabels(Nps);
```
