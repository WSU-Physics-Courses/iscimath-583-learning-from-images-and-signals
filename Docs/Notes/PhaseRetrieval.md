---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.0
kernelspec:
  display_name: Python 3 (math-583)
  language: python
  name: math-583
---

```{code-cell} ipython3
:tags: [hide-cell]

import mmf_setup;mmf_setup.nbinit()
import logging;logging.getLogger('matplotlib').setLevel(logging.CRITICAL)
%matplotlib inline
import numpy as np, matplotlib.pyplot as plt
```

(sec:PhaseRetrieval)=
Phase Retrieval
===============

Consider the following problem: Given a set of images $I_n(x)$ of the form
\begin{gather*}
  I_n(x) = b(x) + a(x)\cos\Bigl(\phi(x) + \theta_n\Bigr) + \eta_n(x),
\end{gather*}
where we assume that $\eta_n(x)$ is noise and relatively small, how can we reconstruct
$\phi(x)$?
This problem is known as [phase retrieval][].  It is common in physical applications of
[interferometry][] or interferometric imaging, and is the subject of research at WSU in
the Engels' and Forbes' groups (see {cite-p}`Mossman:2022` for an application with atom lasers.)

::::{admonition} Try it!

Take a moment and try to implement a solution to this problem for the simple 1D problem
posed in Assignment 1.  To simplify the problem, assume no noise $\eta_n = 0$, and that
we can choose $\theta_n$ as we like -- a useful choice being $\theta_n = 2\pi n/N$ for
$n \in \{0, 1, \dots, N-1\}$ so that averaging $I_n$ over $n$ gives us $b$.  Don't worry
about reconstructing the full phase $\phi$: just find $\phi \in [-\pi/2, \pi/2)$ module
$2\pi$.  To reconstruct the smooth $\phi(x)$, one can use the
[`skimage.restoration.unwrap_phase`](https://scikit-image.org/docs/dev/api/skimage.restoration.html#skimage.restoration.unwrap_phase).
(This last step is a little complicated to implement.)

As encouragement, a simple solution with these conditions can be implemented in less
than 20 lines, which works reasonably well even with moderate noise $\eta_n$.
::::

## A Strategy

Here we present a simple strategy.  In what follows, we will suppress the index $x$:
this may take values in 1D (signals), 2D (images), or any other appropriate space
without altering the ideas.  Just consider these indices raveled together into a single
index so that the images $\ket{I_n}$ can be represented as vectors in an appropriate
high-dimensional space.

A simple strategy is to note that
\begin{gather*}
  \ket{I_n} = \ket{b} + \Bigl(\ket{a\cos\phi}\cos(\theta_n) - \ket{a\sin\phi}\sin(\theta_n)\Bigr) + \ket{\eta_n}.
\end{gather*}
Thus, up to the noise $\eta_n$, the image lives in a 3-dimensional subspace spanned by
the basis vectors
\begin{gather*}
  \ket{I_n} \in \span\bigl\{\ket{b}, \ket{a\cos\phi}, \ket{a\sin\phi}\bigr\}.
\end{gather*}

:::{margin}
**To Do**: Make the nature of this "best" approximation precise.
:::
Our simple strategy will be to package the images as columns of a matrix $\mat{I}$, and
then compute the [SVD][].  The 3 largest singular values and corresponding columns in
$\mat{U}$ will provide the closest approximation to this 3-dimensional subspace.
:::{margin}
Note: numerically, use the incomplete [SVD][] (`full_matrices=True`) to save memory.
:::
\begin{gather*}
  \mat{I} = \begin{pmatrix}\\
    \ket{I_0} & \ket{I_1} & \cdots & \ket{I_{N-1}}\\
    &
  \end{pmatrix}\\
  = \mat{U}\mat{D}\mat{V}^T \approx \sum_{k=0}^{2}\ket{u_k}d_k\!\bra{v_k}.
\end{gather*}
This allows us to construct the subspace
:::{margin}
We conveniently include the singular values $d_k$ with basis vectors so that the images
are constructed from these with coefficients $\bra{v_k}$ which are orthonormal.  We will
see this advantage below.
:::
\begin{gather*}
  \span\bigl\{\ket{b}, \ket{a\cos\phi}, \ket{a\sin\phi}\bigr\} \approx
  \span\bigl\{\ket{u_0}d_0, \ket{u_1}d_1, \ket{u_2}d_2\bigr\}.
\end{gather*}

:::{margin}
This simplification follows from the fact that
\begin{gather*}
  \sum_{n=0}^{N-1} \exp\frac{2\pi \I n}{N}
  = 0.
\end{gather*}
:::
The most general formulation of the problem requires isolating these three components,
but if we have control of $\theta_n$, which is quite a common case, then we can choose
$\theta_n$ such that averaging over $n$ yields $\ket{b}$:
\begin{gather*}
  \ket{b} = \frac{1}{N}\sum_{n=0}^{N-1} \ket{I_n\Bigl(\theta_n = \frac{2\pi n}{N}\Bigr)}.
\end{gather*}
This allows us to analyze instead $\ket{I_n - b} = \ket{I_n} - \ket{b}$ which live in
the 2-dimensional subspace
\begin{gather*}
  \ket{I_n - b} \in \span\bigl\{\ket{a\cos\phi}, \ket{a\sin\phi}\bigr\}.
\end{gather*}

Now, if we have control over $\theta_n$ we can simply evaluate
\begin{gather*}
  \ket{a\cos\phi} = \ket{I_n(\theta_n=0) - b}, \qquad
  \ket{a\sin\phi} = -\ket{I_n(\theta_n=\tfrac{\pi}{2}) - b}.
\end{gather*}
:::{margin}
**To Do**: Explain precisely why.
:::
In the case of noise, however, it is again good to consider the corresponding [SVD][],
which now has only two principle components:
\begin{gather*}
  \mat{I_b} = \begin{pmatrix}\\
    \ket{I_0-b} & \ket{I_1-b} & \cdots & \ket{I_{N-1}-b}\\
    &
  \end{pmatrix}\\
  = \mat{U^b}\mat{D^b}\mat{V^b}^T \approx \sum_{k=0}^{1}\ket{u^b_k}d^b_k\!\bra{v^b_k}.
\end{gather*}
:::{margin}
**To Do**: Explain this better.
:::
Define
\begin{gather*}
  \ket{A} = \ket{ud_0} = \ket{u^b_0}d^b_0, \qquad
  \ket{B} = \ket{ud_1} = \ket{u^b_1}d^b_1.
\end{gather*}
We can now identify
\begin{gather*}
  \ket{A} = \ket{a\cos(\phi + \phi_0)}, \qquad  
  \ket{B} = \ket{a\sin(\phi + \phi_0)},
\end{gather*}
:::{margin}
*This is a slight abuse of the vector notation, but hopefully it is clear.  See the code
for details.*
:::
where $\phi_0$ is a phase.  Thus, we can compute:
\begin{gather*}
  \ket{a} = \sqrt{\ket{A}^2 +\ket{B}^2}, \qquad
  \ket{\phi + \phi_0} = \tan^{-1}\frac{\ket{B}}{\ket{A}}.
\end{gather*}

[phase retrieval]: <https://en.wikipedia.org/wiki/Phase_retrieval>
[interferometry]: <https://en.wikipedia.org/wiki/Interferometry>
[SVD]: <https://en.wikipedia.org/wiki/Singular_value_decomposition>


```python
%matplotlib inline
import numpy as np, matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import InterpolatedUnivariateSpline


class Units:
    ms = 1.0
    micron = 1.0
    m_Rb87 = 1.0  # 86.909184u
    hbar = 0.73073749370249511349  # hbar/(86.909184u/ms*micron^2)
    nK = 0.095668400488336650821  # boltzmann*nK/(86.909184u*micron^2/ms^2)

    mm = 1000 * micron
    m = 1000 * mm
    s = 1000 * ms


u = Units()


class Sim:
    m = u.m_Rb87
    hbar = u.hbar
    Nt = 256
    pixel_size = 1.112 * u.micron
    V_accel = 9.8 * u.m / u.s**2
    width = 38.9 * u.micron
    V0 = 80 * u.nK
    z0 = 89.9 * u.micron
    Lz = 1000.0 * u.micron
    Nz = 256

    # Assume freefall without potential.
    T = np.sqrt(2 * Lz / V_accel)

    def __init__(self, **kw):
        for key in kw:
            if not hasattr(self, key):
                raise ValueError(f"Unknown {key=}")
            setattr(self, key, kw[key])
        self.init()

    def init(self):
        y0 = (0, 0, 0, 0, 0, 0)
        self._species = np.array([1, -1])
        self._res = res = solve_ivp(self.compute_dy_dt,
                                    y0=y0,
                                    t_span=(0, self.T),
                                    max_step=self.T / self.Nt)
        ts = self.ts = res.t
        Nt = len(ts)
        zs, vzs, Ss = self.zs, self.vzs, self.Ss = res.y.reshape((3, 2, Nt))

        #_zs = [InterpolatedUnivariateSpline(ts, _z, k=3) for _z in zs]
        #_vzs = [InterpolatedUnivariateSpline(ts, _v, k=3) for _v in vzs]
        self.Ss_z = [
            InterpolatedUnivariateSpline(_z, _S, k=3)
            for (_z, _S) in zip(zs, Ss)
        ]
        z_top = 20.0
        dz = self.pixel_size
        Nz = int(np.ceil((np.max(zs) - z_top) / dz))
        self.z = z_top + np.arange(Nz) * dz

    def dV(self, z, d=0):
        """Return the differential potential."""
        r = z - self.z0
        if d == 0:
            return self.V0 * np.exp(-2 * r**2 / self.width**2)
        elif d == 1:
            dr_dz = 1
            return -4 * r / self.width**2 * self.V0 * np.exp(
                -2 * r**2 / self.width**2) * dr_dz

    def V(self, z, d=0, species=1):
        if d == 0:
            res = -self.m * self.V_accel * z
        elif d == 1:
            res = -self.m * self.V_accel
        return res + species * self.dV(z, d=d)

    def compute_dy_dt(self, t, y):
        species = np.array([1, -1])
        zs, dzs, Ss = np.reshape(y, (3, 2))
        ddzs = -self.V(zs, d=1, species=self._species) / self.m
        dSs = self.m * dzs**2 / 2 - self.V(zs, d=0,
                                           species=self._species)  # Lagrangian
        return np.ravel([dzs, ddzs, dSs])

    def get_phi(self, z):
        return (self.Ss_z[0](z) - self.Ss_z[1](z)) / self.hbar

    def get_psi(self, z, phase=0):
        # We use the conservative expressions for the weights.
        species = np.array([1, -1])[:, np.newaxis]
        p0 = 0
        ps = np.sqrt(p0**2 + 2 * self.m * (self.V(0, d=0, species=species) -
                                           self.V(z, d=0, species=species)))
        psi_0 = np.exp(1j *
                       (self.Ss_z[0](z) / self.hbar + phase)) / np.sqrt(ps[0])
        psi_1 = np.exp(1j *
                       (self.Ss_z[1](z) / self.hbar - phase)) / np.sqrt(ps[1])
        return psi_0 + psi_1

    def get_n(self, z, phase=0):
        return abs(self.get_psi(z, phase=phase))**2


s = Sim()
z = s.z
phi = s.get_phi(z)
plt.plot(z, s.get_n(z, phase=0))
plt.plot(z, s.get_n(z, phase=2))
```
