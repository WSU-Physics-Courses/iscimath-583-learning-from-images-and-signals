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

import mmf_setup;mmf_setup.nbinit()
import logging;logging.getLogger('matplotlib').setLevel(logging.CRITICAL)
%matplotlib inline
import numpy as np, matplotlib.pyplot as plt
from IPython.display import clear_output, display
```

(sec:Denoising)=
Denoising
=========

:::{margin}
The actual model implemented in the code is slightly different as it needs to include a
regulator in the limit of $p, q \rightarrow 1$.  See {ref}`Code` for details.
:::
Here we discuss the following denoising model:
\begin{gather*}
  E[u] = \int\frac{1}{p}\abs{\vect{\nabla}u}^p
  + \frac{λ}{q}\int\abs{u - d}^q,\\
  E'[u] = -\vect{\nabla}\cdot\Bigl(\vect{\nabla}u \abs{\vect{\nabla}u}^{p-2}\Bigr)
  + λ(u-d)\abs{u-d}^{q-2}.
\end{gather*}

## One Dimension

To better understand what happens, we start by working in 1D.

```{code-cell}
from math_583 import denoise

Nx = 16
x = np.arange(Nx)
u1 = x % 2
u2 = np.where(abs(x-Nx/2) < Nx/4, 1.0, 0)

us = [u1, u2]
ims = [denoise.Image(data=u) for u in us]
ds = [denoise.Denoise(im, sigma=0) for im in ims]

fig, axs = denoise.subplots(len(ims))
for ax, im in zip(axs, ims):
    im.show(im.get_data(), ax=ax)
display(fig)
```

```{code-cell}
mode = "periodic"
pqs = [(2,2), (2,1), (1,2), (1,1)]
lams = [1.0, 0.1, 0.01]
fig, axs = denoise.subplots(len(ims), len(pqs), aspect=2)
args = dict(eps_p=1e-6, eps_q=1e-6, sigma=0, mode=mode)
for row, (p, q) in enumerate(pqs):
    for col, im in enumerate(ims):
        ax = axs[row, col]
        for lam in lams:
            d = denoise.Denoise(im, lam=lam, p=p, q=q, **args)
            u = d.minimize()
            im.show(u, label=f"{lam=:g}", ax=ax)
        ax.legend()
    axs[row, 0].set(title=f"{p=}, {q=}, {mode=}")
clear_output()
display(fig)
```


[DCT]: <https://en.wikipedia.org/wiki/Discrete_cosine_transform>
[DST]: <https://en.wikipedia.org/wiki/Discrete_sine_transform>

[periodic]: <https://en.wikipedia.org/wiki/Periodic_boundary_conditions>
[Dirichlet]: <https://en.wikipedia.org/wiki/Dirichlet_boundary_condition>
[Neumann]: <https://en.wikipedia.org/wiki/Neumann_boundary_condition>
[product rule]: https://en.wikipedia.org/wiki/Product_rule
[Toeplitz]: <https://en.wikipedia.org/wiki/Toeplitz_matrix>
[convolution]: <https://en.wikipedia.org/wiki/Convolution>
[Dirac comb]: <https://en.wikipedia.org/wiki/Dirac_comb>
[Dirac delta function]: <https://en.wikipedia.org/wiki/Dirac_delta_function>
[Kronecker delta]: <https://en.wikipedia.org/wiki/Kronecker_delta>
[FFT]: <https://en.wikipedia.org/wiki/Fast_Fourier_transform>
[FFTw]: <https://fftw.org>
[DFT]: <https://en.wikipedia.org/wiki/Discrete_Fourier_transform>
[machine precision]: <https://en.wikipedia.org/wiki/Machine_epsilon>
[Renormalization Group]: <https://physics-552-quantum-iii.readthedocs.io/en/latest/RenormalizationGroup.html>
[analytic function]: <https://en.wikipedia.org/wiki/Analytic_function>
[ringing artifacts]: <https://en.wikipedia.org/wiki/Ringing_artifacts>
[broadcasting]: <https://numpy.org/doc/stable/user/basics.broadcasting.html>
