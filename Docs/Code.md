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
  E[u] = \int\frac{1}{p}\abs{\vect{\nabla}u}^p \\
  + \frac{λ}{q}\int\abs{u - d}^q,\\
  E'[u] = -\vect{\nabla}\cdot\Bigl(\vect{\nabla}u \abs{\vect{\nabla}u}^{p-2}\Bigr) \\
  + λ(u-d)\abs{u-d}^{q-2}.
\end{multline*}
These are formally equivalent in the continuum limit with vanishing regulators
$ϵ_p, ϵ_q \rightarrow 0$ with an appropriate derivative operator, but numerically
distinct as discussed in {ref}`sec:FT-product`.
:::
This module provides tools to denoise images based on the following model:
\begin{gather*}
  E[u] = \int\frac{1}{p}\abs{\vect{\nabla}u}^p + \frac{λ}{q}\int\abs{u - d}^q.
\end{gather*}
Minimizing is equivalent to solving:
\begin{gather*}
  E'[u] = -\nabla^2u + λ(u - d) = 0.
\end{gather*}

A direct approach to the minimization procedure is given by {meth}`Denoise.minimize`
which uses {func}`scipy.optimize.minimize` and the [`L-BFGS-B`](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html#optimize-minimize-lbfgsb) method).  Minimization is not the fastest, but is quite
general, and be directly applied to more complicated methods.

\begin{gather*}
  E[u] = \int\frac{1}{p}\bigl(\abs{\vect{\nabla}u}^2 + ϵ_p\bigr)^{p/2} 
       + \frac{λ}{q}\int\bigl((u - d)^2 + ϵ_q\bigr)^{q/2},\\
  E'[u] = -\vect{\nabla}\cdot\Bigl(
    (\vect{\nabla} u) \bigl(
      \abs{\vect{\nabla}u}^2 + ϵ_p
    \bigr)^{(p-2)/2}
  \Bigr) 
  + λ(u - d)\bigl((u - d)^2 + ϵ_q\bigr)^{(q-2)/2}\\
\end{gather*}

:::{admonition} Details about anti-symmetric derivatives $\mat{D}^T = - \mat{D}$.
:class: dropdown

In order to obtain the expression for $E'[u]$, we need to compute
\begin{gather*}
  \pdiff{\bigl[\abs{\vect{\nabla}u}^2\bigr]_{a}}{u_b}
\end{gather*}
which appears in the first integral after the chain rule is applied.  Noting that
$\vect{\nabla}$ is a linear operator, we will have a matrix representation
$\vect{\mat{D}}_{ac}$ such that
\begin{gather*}
  \bigl[\abs{\vect{\nabla} u}^2\bigr]_{a} = 
  \sum_{i, c, d}D^{i}_{ac}u_c D^{i}_{ad}u_d\\
  \pdiff{\bigl[\abs{\vect{\nabla}u}^2\bigr]_{a}}{u_b}
  = \sum_{i, d}D^{i}_{ab} D^{i}_{ad}u_{d} 
  + \sum_{i, c}D^{i}_{ac}u_{c} D^{i}_{ab}
  = 2\sum_{i, c}D^{i}_{ab} D^{i}_{ac}u_{c}.
\end{gather*}
When included in the integral with an additional factor $f_{a} =
  \bigl(\abs{\vect{\nabla}u}^2 + ϵ_p\bigr)^{(p-2)/2}_{a}$ we end up with
\begin{gather*}
  2\sum_{i, a, c}D^{i}_{ab} D^{i}_{ac}u_{c} f_{a}
  = 2 \sum_{a} \underbrace{(-D^{i}_{ba})}_{D^{i}_{ab}}\Bigl(\sum_{i, c} D^{i}_{ac}u_{c} f_{a}\Bigr)
  \equiv - 2 \vect{\nabla}\cdot\Bigl((\vect{\nabla}u) f\Bigr),
\end{gather*}
iff the matrix representations of the derivatives are anti-symmetric
\begin{gather*}
  \mat{D}^{\dagger} = - \mat{D}, \qquad
  D^{i}_{ab} = - D^{i}_{ba}.
\end{gather*}
Alternatively, the outer derivative must be computed using the transpose of the original
derivative matrix as in the explicit formula. 
*Note: In the above, we have not consider complex values.  Similar formulae apply with
appropriate complex conjugations applied.*
:::

:::{admonition} Computing $\vect{\nabla}$ and $\nabla^2$.

Hi
:::

A faster approach is provided by {meth}`Denoise.solve`, which directly solves the
minimization condition using {ref}`sec:FourierTechniques`.

```{eval-rst}
.. automodule:: math_583.denoise
    :members:
```
