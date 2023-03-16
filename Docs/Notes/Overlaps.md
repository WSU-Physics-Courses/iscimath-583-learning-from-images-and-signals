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

(sec:Overlaps)=
Overlaps
========

Various algorithms like matching pursuit work to find a signal by maximizing the "overlap"
between a template and the data.  The idea behind this that the noise will tend to
cancel from this overlap, leaving a good match if the template has the same form as the
signal.  Here we demonstrate some simple examples and discuss the meaning of this.

Two examples from class include:

1. Finding the frequency $p=k$ of a periodic signal:

   \begin{gather*}
     f(x) = f_0 + a\cos(kx + \phi).
   \end{gather*}
      
2. Finding the location $p=x_0$ of a step function:

   \begin{gather*}
     f(x) = f_0 + aH(x - x_0), \qquad 
     H(x) = \begin{cases}
       0 & x < 0,\\
       1 & x \geq 0.
     \end{cases}
   \end{gather*}

In both cases, there is a single parameter $p$ of interest -- $k$ and $x_0$ respectively
here -- and some nuisance parameters -- $f_0$, $a$, and $\phi$.

The idea is to use as an objective function $E(p)$, the overlap between the signal $u(x) =
f(x)+ η$ where $η$ is noise, and some template $T_{p}(x)$ that depends only on the
parameter $p$ of interest.  This reduces the problem to a single dimension where one can
exhaustively explore the options.  For this to work, the overlap should be maximized at
the optimal value of the desired parameter, and should be insensitive to the nuisance
parameters.

To simplify our notation, we represent the noisy signal $\ket{u}$ and the
template $\ket{T(p)}$ as vectors so that the overlap is
\begin{gather*}
  E(p) = \abs{\braket{T(p)|u}}.
\end{gather*}

Our strategies are based mostly on greedy approaches of finding $p_0$:
\begin{gather*}
  \max_{p} E(p).
\end{gather*}

## Remove the Means

To minimize the effects of the offset $f_0$ and to maximize the overlap, it is usually a
good idea to subtract the mean from both the data and the template.  Here we
demonstrate:

```{code-cell}
def H(x):
    return np.where(x < 0, 0, 1)

rng = np.random.default_rng(seed=2)

f0 = 1.2
a = 0.7
x0 = -0.5
sigma = 1

# Get the data
x = np.linspace(-1, 1, 500)
u0 = f0 + a * H(x - x0)
u = u0 + rng.normal(scale=sigma, size=x.shape)
u_ = u - u.mean()

# Here we make the template as a matrix.  This is faster
# but you can run out of memory if you have lots of points.
# Dask might help, or you can loop.
x0s = np.linspace(-1, 1, 201)
T = H(x[None, :] - x0s[:, None])
T_ = T - T.mean(axis=-1)[:, None]

fig, axs = plt.subplots(2, 1, gridspec_kw=dict(height_ratios=(1,2)), sharex=True)
ax = axs[0]
ax.plot(x, u, label='data')
ax.plot(x, u0, label='signal')
ax.legend()
ax.set(xlabel='$x$', ylabel='$u$')

ax = axs[1]
Es = abs(T.dot(u_))
ax.plot(x0s, Es, label=r"$u-\bar{u}$")

Es = abs(T_.dot(u))
ax.plot(x0s, Es, label=r"$T-\bar{T}$")

Es = abs(T_.dot(u_))
ax.plot(x0s, Es, label=r"$u-\bar{u}$ and $T-\bar{T}$")

ylim = ax.get_ylim()

Es = abs(T.dot(u))
ax.plot(x0s, Es/Es.max()*max(ylim), label="No subtraction (scaled)")
ax.axvline([x0], c='y', ls='--')
ax.legend()
ax.set(xlabel='$x_0$', ylabel='overlap $E(x_0)$', ylim=ylim);
```

:::{admonition} Do it! Show why subtracting either mean is sufficient?
:class: dropdown

Consider subtracting both means:
\begin{gather*}
  \braket{T-\bar{t}|u-\bar{u}}
  = \braket{T|u-\bar{u}} - \bar{t}\overbrace{\braket{u-\bar{u}}}^{0}
  = \braket{T-\bar{t}|u} - \bar{u}\overbrace{\braket{T-\bar{T}}}^{0}.
\end{gather*}

Hence, one can choose to subtract the mean from either the template, or the data.
:::

Thus, by subtracting the mean, we remove dependence on the parameter $f_0$.
Furthermore, since the we only need to find the maximum value, we are insensitive to the
overall amplitude $a$ of the function.  This gives us the desired dimensional reduction
to a single parameter $x_0$.

## Matching a Sine Wave

The problem of finding the best match to a sine wave has three nuisance parameters.
Subtracting the mean $f_0$ and finding the maximum will again remove dependence on $f_0$
and $a$, but the phase $\phi$ poses an additional problem.

Here we can solve this by noting that
\begin{gather*}
  \cos(kx+𝜙) = \cos 𝜙 \cos(kx) - \sin 𝜙 \sin(kx).
\end{gather*}
Thus, we can find $𝜙$ by matching two templates $\cos(kx)$ and $\sin(kx)$.  The precise
implementation is a little tricky because, depending on the sample points $x_i$ and the
size of the domain, these two templates might not be orthogonal.  To deal with this, we
do a little linear algebra, that is reminiscent of the [Gram-Schmidt][].  We want two
templates $\ket{c}$ and $\ket{s}$ such that, for a signal $\ket{u} = a\ket{\cos(kx)} + b
\ket{\sin(kx)}$ we have
\begin{gather*}
  a = \braket{c|u}, \qquad b = \braket{s|u}.
\end{gather*}
We simply organize these as a matrix equation
\begin{gather*}
  \ket{u} =
  \overbrace{
  \begin{pmatrix}
    \vphantom{\Big|}
    \ket{\cos(kx)} & \ket{\sin(kx)}
  \end{pmatrix}}^{\mat{M}^{(N×2)}}
  \begin{pmatrix}
    a \\
    b
  \end{pmatrix},
  \\
  \begin{pmatrix}
    a \\
    b
  \end{pmatrix}
  =
  \overbrace{
  \begin{pmatrix}
    \bra{c} \\
    \bra{s}
  \end{pmatrix}}^{\mat{T}^{(2×N)}}
  \ket{u}
  =
  \mat{T}\mat{M}
  \begin{pmatrix}
    a \\
    b
  \end{pmatrix}.
\end{gather*}
In other words, our desired template $\mat{T}$ is simply a [pseudoinverse][] of the matrix
$\mat{M}$:
\begin{gather*}
  \begin{pmatrix}
    \bra{c} \\
    \bra{s}
  \end{pmatrix}
  = \mat{T} 
  = \mat{M}^{-1}
  = \begin{pmatrix}
    \vphantom{\Big|}
    \ket{\cos(kx)} & \ket{\sin(kx)}
  \end{pmatrix}^{-1}.
\end{gather*}
:::{margin}
Be careful to use the [`mode='reduced'`
version](https://numpy.org/doc/stable/reference/generated/numpy.linalg.qr.html) or the
equivalent, otherwise $\mat{Q}$ will be $N×N$ with $N-2$ useless columns.
:::
This can be easily computed by in terms of the reduced [QR decomposition][]:
\begin{gather*}
  \mat{M}^{(N×2)} = \mat{Q}^{(N×2)}\mat{r}^{(2×2)}, \qquad
  \mat{T}^{(2×N)} = \mat{r}^{-1}\mat{Q}^T.
\end{gather*}
Here $\mat{Q}$ contains two orthonormal columns $\mat{Q}^T\mat{Q} = \mat{1}$, and
$\mat{r}$ is an upper-triangular $2×2$ matrix that is easily inverted.

```{code-cell}
N = 500
x = np.linspace(0, 10, N)
k = 3.0
M = np.array([
    np.cos(k*x),
    np.sin(k*x)]).T
assert M.shape == (N, 2)

Q, r = np.linalg.qr(M, mode='reduced')
assert Q.shape == (N, 2)
assert r.shape == (2, 2)

T = np.linalg.solve(r, Q.T)  # Preferred to inverting r explicitly.
assert np.allclose(np.linalg.inv(r) @ Q.T, T)
assert T.shape == (2, N)
#%timeit np.linalg.solve(r, Q.T)
#%timeit np.linalg.inv(r) @ Q.T
```

We use it like this, with an overlap value of $\sqrt{a^2 + b^2}$:
```{code-cell}
phi = 1.2
u = np.cos(k*x + phi)
a, b = T @ u
assert np.allclose(u, a*np.cos(k*x) + b*np.sin(k*x))
assert np.allclose(np.arctan2(-b, a), phi)

def get_overlap(k, u, x=x):
    Q, r = np.linalg.qr(np.array([np.cos(k*x), np.sin(k*x)]).T, mode='reduced')
    a, b = np.linalg.solve(r, Q.T @ u)
    return np.sqrt(a**2+b**2)
```

Here we try to find a signal in the noise:

```{code-cell}
rng = np.random.default_rng(seed=2)

f_0 = 0.2
a = 0.3
phi = 1.5
k = 5.4
sigma = 1

# Get the data
N = 500
x = np.linspace(0, 10, N)
u0 = f_0 + a*np.cos(k*x + phi)
u = u0 + rng.normal(scale=sigma, size=x.shape)
u_ = u - u.mean()

fig, axs = plt.subplots(2, 1, gridspec_kw=dict(height_ratios=(1, 2)))
ax = axs[0]
ax.plot(x, u, label='data')
ax.plot(x, u0, label='signal')
ax.legend()
ax.set(xlabel='$x$', ylabel='$u$')

Nk = 500
ks = np.linspace(0, 3*k, Nk)[1:] # Skip k=0
Es = [get_overlap(k, u=u_, x=x) for k in ks]

ax = axs[1]
ax.plot(ks, Es, label=r"$u-\bar{u}$")

Es = [get_overlap(k, u=u, x=x) for k in ks]
ax.plot(ks, Es, '-C1', label="No subtraction")
ax.axvline([k], c='y', ls='--')
ax.legend()
ax.set(xlabel='$k$', ylabel='overlap $E(k)$');
```
Notice that subtracting the mean here does not help as much as before, especially for
small $k$.  This is because in our current formulation, the inversion of $\mat{r}$
becomes singular when $k=0$ due to the fact that $\sin(0x) = 0$.

:::{admonition} Do it!  How can you fix this? (Nontrivial)
:class: dropdown

Below we demonstrate that, for real signals, we can use a complex template that works
very well and solves this problem.  You should be able to work backwards to see how this
works for the real overlap case.
:::

If your signal is real, then we can use complex numbers to simplify things a bit:
\begin{gather*}
  \bra{T} = \bra{c} - \I\bra{s}.
\end{gather*}

```{code-cell}
def get_overlap_complex(k, u, x=x):
    return abs(np.exp(-1j*k*x) @ u)

fig, axs = plt.subplots(2, 1, gridspec_kw=dict(height_ratios=(1, 2)))
ax = axs[0]
ax.plot(x, u, label='data')
ax.plot(x, u0, label='signal')
ax.legend()
ax.set(xlabel='$x$', ylabel='$u$')

Nk = 500
ks = np.linspace(0, 3*k, Nk)[1:] # Skip k=0
Es = [get_overlap_complex(k, u=u_, x=x) for k in ks]

ax = axs[1]
ax.plot(ks, Es, label=r"$u-\bar{u}$")

Es = [get_overlap_complex(k, u=u, x=x) for k in ks]
ax.plot(ks, Es, '-C1', label="No subtraction")
ax.axvline([k], c='y', ls='--')
ax.legend()
ax.set(xlabel='$k$', ylabel='overlap $E(k)$');
```


[QR decomposition]: <https://en.wikipedia.org/wiki/QR_decomposition>
[Gram-Schmidt]: <https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process>
[pseudoinverse]: <https://en.wikipedia.org/wiki/Generalized_inverse>


## Optimal Template?

One might consider, what is the "optimal" template?

\begin{gather*}
  \braket{T(p)|f(x_0, a, f_0) + η}
\end{gather*}






