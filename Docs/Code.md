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
```

(sec:code)=
# Code

Here we describe the various tools implemented in the {mod}`math_583` package.

## {mod}`math_583`

The module provides the following classes:

```{eval-rst}
.. automodule:: math_583
    :members:
```

## {mod}`math_583.denoise`

```{py:module} math_583.denoise
```

:::{margin}
This model is a proxy for
\begin{multline*}
  E[u] = \int\frac{1}{2}\abs{\vect{\nabla}u}^2 \\
  + \frac{\lambda}{2}\int\abs{u - d}^2.
\end{multline*}
These are formally equivalent in the continuum limit if the boundary conditions vanish, but
numerically distinct as discussed in {ref}`sec:FT-product`.
:::
This module provides tools to denoise images based on the following model:
\begin{gather*}
  E[u] = \int\frac{-u\nabla^2 u}{2} + \frac{\lambda}{2}\int\abs{u - d}^2.
\end{gather*}
Minimizing is equivalent to solving:
\begin{gather*}
  E'[u] = -\nabla^2u + \lambda(u - d) = 0.
\end{gather*}

A direct approach to the minimization procedure is given by {meth}`Denoise.minimize`
which uses {func}`scipy.optimize.minimize` and the [`L-BFGS-B`](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html#optimize-minimize-lbfgsb) method).  Minimization is not the fastest, but is quite
general, and be directly applied to more complicated methods.

A faster approach is provided by {meth}`Denoise.solve`, which directly solves the
minimization condition using {ref}`sec:FourierTechniques`.

```{eval-rst}
.. automodule:: math_583.denoise
    :members:
```




