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

(sec:FourierTechniques)=
Fourier Techniques
==================

The idea behind Fourier techniques is to express a signal $f(x)$ in terms of a series of sine
waves, or often more conveniently, in terms of plane waves $\exp(\I k x)$:
\begin{gather*}
  f(x) = \sum_{k} \tilde{f}_{k}e^{\I k x}, \qquad
  e^{\I k x} = \cos(kx) + \I \sin(kx).
\end{gather*}
The coefficients $\tilde{f}_{k}$ are called the Fourier coefficients, and the role of
the Fourier transform is to compute these.

Exactly what this means depends on the context, which determines which values of $k$
appear in the preceding expressions.  Some common cases include:

* **Fourier Transform:** The most general expression for a function $f(x):
  \mathbb{R}\rightarrow \mathbb{C}$ is the continuous Fourier transform:
  \begin{gather*}
    \newcommand{\F}{\mathcal{F}}
    \overbrace{
      f(x) = \underbrace{
        \int_{-\infty}^{\infty} \frac{\d{k}}{2\pi}\; \tilde{f}_{k} e^{\I k x}
      }_{\text{Inverse Fourier transform}}}^{f = \F^{-1}(\tilde{f}),} 
      \qquad
    \overbrace{
      \tilde{f}_{k} = \underbrace{
        \int_{-\infty}^{\infty} \d{x}\; e^{-\I k x} f(x)
      }_{\text{Fourier transform}}}^{\tilde{f} = \F(f).}
  \end{gather*}
  :::{margin}
  This can be used to express the completeness relationship in terms of the following
  integrals:
  \begin{gather*}
    \int_{-\infty}^{\infty}\d{x}\; e^{\I (k-q) x} = 2\pi \delta(k-q),\\
    \int_{-\infty}^{\infty}\frac{\d{k}}{2\pi}\; e^{\I k (x-y)} = \delta(x-y).
  \end{gather*}
  :::
  The key for this is the following **completeness relationship**:
  \begin{gather*}
    \int_{-\infty}^{\infty}\d{x}\; e^{\I k x} = 2\pi \delta(k).
  \end{gather*}
  where $\delta(x)$ is the [Dirac delta function][].
 
* **Fourier Series:** For strictly periodic functions $f(x+L) = f(x)$, we must choose
  $k$ so that the basis functions are periodic.
  \begin{gather*}
    e^{\I k (x + L)} = e^{\I k x}e^{\I k L} = e^{\I k x}, \qquad
    k_n = \frac{2\pi n}{L}.
  \end{gather*}
  Hence, the $k$s take on discrete values and the function $f(x)$ can be expressed in
  terms of a Fourier series:
  \begin{gather*}
    f(x) = \sum_{n=-\infty}^{\infty} \tilde{f}_{k_n} e^{\I k_n x}.
  \end{gather*}
  The corresponding completeness relationship is
  \begin{gather*}
    \sum_{n=-\infty}^{\infty} e^{\I k_n x} = \sum_{m=-\infty}^{\infty} \delta(x - mL)
    \equiv Ш_{L}(x).
  \end{gather*}
  The right-hand side here is sometimes known as a [Dirac comb][].
  
* **Discrete Fourier Transform:** If we further restrict our attention to functions
  sampled on a set of lattice of points $x_m = x_0 + m \delta$ with lattice space
  $\delta = L/N$, then we can restrict the number of Fourier modes to be finite:
  \begin{gather*}
    f(x_m) = \sum_{n=0}^{N-1} \tilde{f}_{k_n} e^{\I k_n x_m}.
  \end{gather*}
  :::{margin}
  This is "obvious" if you think of $e^{\I \theta}$ as unit vectors in the complex
  plane.  This sums over a set of vectors that completely cancels unless the exponent is
  zero, in which case they all point in the same direction.
  :::
  The key for this is the following **completeness relationship**:
  \begin{gather*}
    \frac{1}{N}\sum_{m=0}^{N-1} \exp\left(\frac{2\pi \I n m}{N}\right) = \delta_{n0},
  \end{gather*}
  where $\delta_{mn}$ is the [Kronecker delta][].  This discrete Fourier transform (DFT)
  is the domain of computer implementations of the [FFT][].

```{code-cell}
N = 7
n = np.arange(N)[:, None]
m = np.arange(N)[None, :]
M = np.exp(2j*np.pi * m * n / N) 
assert np.allclose(M.sum(axis=1)/N, [1, 0, 0, 0, 0, 0, 0])

phasor = np.cumsum(M, axis=1)
fig, ax = plt.subplots()
for n in range(N):
    z = phasor[n]
    ax.plot(z.real, z.imag, label=f"{n=}")
    for z0, z1 in zip(z[:-1], z[1:]):
        dz = z1 - z0
        ax.arrow(z0.real, z0.imag, dz.real, dz.imag, width=0.05,
        length_includes_head=True, fc=f"C{n}", ec=f"C{n}")
ax.set(aspect=1)
ax.legend(loc='upper right');
```

```{margin}
Note: , for performance, numerical implementations shift the wavenumber integer $n$ to lie
in
\begin{gather*}
  n \in \{\}
\end{gather*}
```

## A Unified Approach
I recommend considering the $N$-point discrete Fourier transform as the base case, and
then taking the appropriate continuum ($N\rightarrow \infty$) or thermodynamic ($L
\rightarrow \infty$) limits:
\begin{gather*}
  n \in \{0, 1, \cdots, N-1\},\\
  \begin{aligned}
    x_{n} &= n \d{x} \in [0, L) \mod L, &
    \d{x} &= \frac{L}{N},\\
    k_{n} &= n\d{k} \in \Bigl[\frac{\pi}{\d{x}}, \frac{\pi}{\d{x}}\Bigr) \mod \frac{2\pi}{\d{x}}, &
    \d{k} &= \frac{2\pi}{L}.
  \end{aligned}
\end{gather*}
Here the momenta $k_n$ shifted by appropriate factors of $2\pi/L$ to lie in the interval
$[-\pi/\d{x}, \pi/\d{x})$.  Thus, numerically, the values of the frequencies $f_n$
returned by {py:func}`numpy.fft.fftfreq` for example are given by the code
```python
f_n = ((np.arange(N) / N + 0.5) % 1 - 0.5)/dx
```
```{code-cell}
:tags: [hide-cell]

L = 1.2
for N in range(3, 11):
    dx = L / N
    f_n = ((np.arange(N) / N + 0.5) % 1 - 0.5) / dx
    assert np.allclose(f_n, np.fft.fftfreq(N, dx))
```

With these definitions, we can define the continuum version by taking $N\rightarrow
\infty$ and/or $L\rightarrow \infty$ while taking
\begin{align*}
  \d{x}\sum_{n} = \frac{L}{N}\sum_{n} &\rightarrow \int \d{x}, \\
  \d{k}\sum_{n} = \frac{2\pi}{L}\sum_{n} &\rightarrow \int \d{k}, \\
  \delta_{mn}/\d{x} &\rightarrow \delta(x_m -x_n),\\
  \delta_{mn}/\d{k} &\rightarrow \delta(k_m - k_n).
\end{align*}
I generally like to keep my factors of $2\pi$ with my momentum integrals and momentum
delta-functions, so I also use
\begin{align*}
  \frac{1}{L}\sum_{n} &\rightarrow \int \frac{\d{k}}{2\pi}, &
  L\delta_{mn} &\rightarrow 2\pi\delta(k_m - k_n).
\end{align*}
:::{margin}
Notice that the top relationship is purely numerical: all factors of length or momentum,
frequency and time, etc. cancel.  This is what is implemented numerically by the DFT, up
to the overall factor of $1/N$ which sometimes is included, but may be dropped for
performance (e.g. with the
[FFTw](https://fftw.org/fftw3_doc/The-1d-Discrete-Fourier-Transform-_0028DFT_0029.html).)
Check your documentation for details.
:::
The completeness relationship follows from:
\begin{align*}
  \frac{1}{N}\sum_{n=0}^{N-1} e^{\overbrace{2\pi \I mn/N}^{\I k_m x_n}} &= \delta_{m0},\\
  \underbrace{\frac{L}{N}}_{\d{x}}\sum_{n=0}^{N-1} e^{\I x_n k_m} 
  \rightarrow \int \d{x}\; e^{\I x k_m} &= 2\pi \delta(k_m) \leftarrow L\delta_{m0},\\
  \underbrace{\frac{1}{L}}_{\d{k}/2\pi}\sum_{n=0}^{N-1} e^{\I x_n k_m} 
  \rightarrow \int \frac{\d{k}}{2\pi}\; e^{\I x k_m} &= \delta(x_m) \leftarrow \frac{\delta_{m0}}{\d{x}},\\
\end{align*}

::::{admonition} The Fast Fourier Transform ([FFT][]).
The discovery of the Fast Fourier Transform ([FFT][]) algorithm revolutionized many
computational techniques by dropping the cost of computing the discrete Fourier transform from
$\order(N^2)$ to $\order(N\log N)$.  Current implementations (see the [FFTw]) keep the
prefactor small, making [FFT][]-based techniques preferred wherever applicable.  In
particular, the [FFT][] has efficient implementations, both on CPUs and on hardware
accelerators like GPUs.  Thus, if the computational cost of an algorithm is dominated by
the DFT, then these algorithms can be implemented almost as efficiently in a high-level
language like Python as in a low-level, high-performance language like C++ or Fortran.
::::

## Derivatives

One of the primary uses of Fourier techniques is to compute derivatives:
\begin{align*}
  f(x) &= \sum_{n} \tilde{f}_{k_n} e^{\I k_n x},\\
  f'(x) &= \sum_{n} \I k_n \tilde{f}_{k_n} e^{\I k_n x},\\
  f''(x) &= \sum_{n} -k_n^2 \tilde{f}_{k_n} e^{\I k_n x},\\
  f^{(d)}(x) &= \sum_{n} (\I k_n)^d \tilde{f}_{k_n} e^{\I k_n x},\\
\end{align*}
:::{margin}
If you are familiar with quantum mechanics, you might recall that the momentum operator
$p = \hbar k$ behaves like
\begin{gather*}
  p \equiv -\I\hbar\pdiff{}{x}.
\end{gather*}
:::
Thus, "in Fourier space", we have:
\begin{gather*}
  \diff{}{x} \equiv \I k
\end{gather*}
in the sense that the Fourier coefficients are multiplied by this factor $\tilde{f}_k \rightarrow \I
k \tilde{f}_k$.

We used this in our code to quickly implement the Laplacian operator for images
\begin{gather*}
  \nabla^2 = \pdiff[2]{}{x} + \pdiff[2]{}{y}:
\end{gather*}

```{code-cell}
def laplacian(f, dx=1.0):
    """Return the Laplacian of f."""
    
    # If f.shape is fixed, K2 could be precomputed for speed
    Nx, Ny = f.shape
    Lx, Ly = dx*Nx, dx*Ny
    kx, ky = [2*np.pi * np.fft.fftfreq(_N, dx) for _N in (Nx, Ny)]
    
    # x goes down (first index), y goes across (second index)
    kx, ky = kx[:, np.newaxis], ky[np.newaxis, :]
    
    K2 = kx**2 + ky**2
    
    return np.fft.ifftn(-K2 * np.fft.fftn(f))
```

```{code-cell}
from scipy.ndimage import laplace
from math_583 import denoise
im = denoise.Image()
f = im.get_data()
d2f = laplace(f, mode='wrap')
d2f_fft = laplacian(f)
assert np.allclose(d2f_fft.imag, 0)
d2f_fft = d2f_fft.real

rel_err = abs(d2f_fft - d2f).mean() / abs(d2f).mean()
print(f"{rel_err=:.2g}")

fig, axs = denoise.subplots(2)
im.show(d2f, vmin=-0.1, vmax=0.1, ax=axs[0])
axs[0].set(title="Finite Difference")
im.show(d2f_fft, vmin=-0.1, vmax=0.1, ax=axs[1])
axs[1].set(title="FFT");
```

Although these look quite similar, the actual error is large in places because the
function {py:func}`scipy.ndimage.laplace` uses a finite difference approximation with a
3-point stencil:
\begin{gather*}
  f''(x) \approx \frac{f(x-h) - 2f(x) + f(x+h)}{h^2} = D_2[f](x).
\end{gather*}
To understand the precise nature of the difference between this an the FFT method, we
can also apply a Fourier technique:
\begin{align*}
  D_2[f] &= \sum_{n} \tilde{f}_{k_n} \frac{
    \overbrace{
      e^{\I k_n (x-h)} - 2e^{\I k_n x}  + e^{\I k_n (x+h)}
    }^{e^{\I k_n x}(e^{-\I k_n h} - 2  + e^{\I k_n h})}
  }{h^2}
  = \sum_{n} \tilde{f}_{k_n} e^{\I k_n x}
  2\frac{
    \overbrace{\cos(k_n h) - 1}^{\frac{(k_n h)^2}{2!} - \frac{(k_n h)^4}{4!} + \cdots}
  }{h^2}\\
  &= \sum_{n} \tilde{f}_{k_n} e^{\I k_n x}
  \left(
    -k_n^2 +\frac{k_n^4 h^2}{12} + \order(h^4)
  \right).
\end{align*}

Thus, we see that the finite difference approximation has the same $-k^2$ term, but
contains additional terms that vanish in the $h\rightarrow 0$ limit.  In our case,
$h=1$, $L\approx N \approx 500$, and so the maximum momentum is $k_\max \approx \pi$.
We expect the relative error to thus be
\begin{gather*}
  \frac{k_\max^2 h^2}{12} \approx \frac{\pi^2}{12} \approx 0.8,
\end{gather*}
consistent with the numerical value.  If the image were smooth, then $k_\max$ would be
smaller than this, reducing the error, but, since there are sharp features, the relative
error here is large.

```{code-cell}
def laplacian2(f, dx=1.0):
    """Return the Laplacian of f using the cos() dispersion."""
    Nx, Ny = f.shape
    Lx, Ly = dx*Nx, dx*Ny
    kx, ky = [2*np.pi * np.fft.fftfreq(_N, dx) for _N in (Nx, Ny)]
    kx, ky = kx[:, np.newaxis], ky[np.newaxis, :]
    h = dx
    K2 = -2 * ((np.cos(kx*h) - 1) + (np.cos(ky*h) - 1)) / h**2
    
    return np.fft.ifftn(-K2 * np.fft.fftn(f))

from scipy.ndimage import laplace
from math_583 import denoise
d2f_fft = laplacian2(f)
assert np.allclose(d2f_fft.imag, 0)
d2f_fft = d2f_fft.real

rel_err = abs(d2f_fft - d2f).mean() / abs(d2f).mean()
print(f"{rel_err=:.2g}")
```

From this tiny error -- on the order of the [machine precision][] -- we see that these
are indeed equivalent.

::::{admonition} The Connection with Renormalization Group
:class: dropdown

The picture here is closely related to the [Renormalization Group][].  In Fourier
space, there are infinitely many different forms of the second derivative operator
\begin{gather*}
  \diff[2]{}{x} \equiv -k^2 + c_2 hk^3 + c_4 h^2k^4 + \cdots,
\end{gather*}
where the coefficients $c_n$ are dimensionless.  As one "coarse grains", taking the
lattice spacing $h \rightarrow 0$, all of these different derivative operators approach
the same continuum result.  This requires that the momenta $k$ be bounded:
i.e. that there is some smallest natural length scale $\xi$ below which $h \ll \xi$ the
physics does not change.  This is the so-called "continuum limit", and any sensible
computational model should have a well-defined limit.

In this case, the image or signal is often characterized by an [analytic function][] --
functions whose power-series converges everywhere -- and the Fourier technique gives
exponential accuracy.
:::{margin}
In lattice gauge theories, such corrections lead to so-called "improved actions".
:::
Approximations with exponential convergence like this are often
called spectral methods (see {cite:p}`Boyd:1989`), and discretization methods will often
add correction terms to improve convergence.  In this sense, the Fourier approach
with just the $-k^2$ term may be considered "the best", but it is far from clear that
this is "the best" from a computational perspective.

For example, an image might actually have sharp features.  These will not be described
by [analytic functions][analytic function], and spectral methods will be no better than
finite-difference techniques, with power-law convergence.  Finite-difference techniques
can have an advantage in these cases, minimizing [ringing artifacts][], for example.
Don't immediately give up on less-accurate approximations unless you know something
about the smoothness of your signals.

::::

### Convolution and Toeplitz Matrices

In the previous section, we saw that Fourier techniques allowed us to obtain a spectral
representation of the derivative, and to understand how the finite difference formula
behaves.  The reason this works is because these operations behave as a [convolution][]
in real space, which becomes multiplication in Fourier space:
\begin{gather*}
  (f*g)(x) = \int_{-\infty}^{\infty} f(x-z)g(z)\d{z},\qquad
  \F(f*g) = \F(f)\F(g).
\end{gather*}

The convolution, in turn, is simply matrix multiplication by a matrix whose entries
depend only on the distance from the diagonal -- so-called [Toeplitz][] matrices.
Finite-difference matrices have this form (with appropriate boundary conditions --
periodic as shown, or Dirichlet):
\begin{gather*}
  \mat{D}_2 = \frac{1}{\d{x}^2}
  \begin{pmatrix}
  -2 & 1 &       &    & 1\\
  1 & -2 & 1       \\
    & 1 & \ddots & \ddots\\
    &   & \ddots & -2 & 1\\
  1 &   &        & 1  & -2.
  \end{pmatrix}.
\end{gather*}
To compute the derivative, take the Fourier transform of your function, multiply by the
Fourier transform of the [Toeplitz][] matrix, and then take the inverse transform:
\begin{gather*}
  f'(x) \approx \underbrace{\F^{-1}\Bigl(-k^2 \F(f)\Bigr)}_{\text{Fourier}}
        \approx \underbrace{
          \F^{-1}\Bigl(2(\cos k - 1)\F(f)\Bigr)
        }_{\text{finite difference}},
\end{gather*}
etc.

If your signal processing algorithm can be expressed as a [Topelitz][] matrix, then
Fourier techniques provide a simple way of exactly solving the problem (up to round-off
errors) as the matrix will be diagonal in Fourier space.

### Product rule

It is important to note that the [product rule][] -- an important mathematical identity
-- generally does not hold for numerical derivative techniques.  Specifically, for two
functions $f(x)$ and $g(x)$, while
\begin{gather*}
  \diff{f(x)g(x)}{x} = f'(x)g(x) + f(x) g'(x),
\end{gather*}
the corresponding case does not hold numerically:
\begin{gather*}
  D(fg) \approx D(f)g + fD(g).
\end{gather*}
While this should approximately hold for smooth functions, it will almost always have
errors for functions with sharp boundaries.  To see this, consider a matrix
representation:
\begin{gather*}
  [D(f)]_{i} = \sum_j D_{ij}f_j, \qquad
  [D(fg)]_i = \sum_j D_{ij}f_jg_j, \\
  [D(f)g + fD(g))]_i = \sum_j D_{ij}(f_jg_i + f_ig_j).
\end{gather*}
:::{margin}
This is the worst case where $f$ is as "sharp as possible", requiring that $g$ be
completely flat.  If $f$ is smooth, then the product rule can hold with high accuracy
for smooth, but varying $g$.
:::
Now consider $f_i = \delta_{ia}$ and let $g'_i \equiv [D(g)]_i$.  Then, if the product
rule was exactly satisfied, we would have
\begin{gather*}
  D_{ia}g_a = D_{ia}g_i + \delta_{ai}g'_i.
\end{gather*}
This says that, if $i=a$, then $g'_a = 0$: hence the relationship can only be exactly
true if $g$ must be a constant.

Care must be employed when using the product rule in finite manipulations,
especially integration by parts.  Thus, while the following is correct in the continuum
limit (up to boundary terms which vanish for [periodic][], [Dirichlet][], or [Neumann][]
boundary conditions):
\begin{gather*}
  E[u] = \int\frac{1}{2}\abs{\vect{\nabla}u}^2 + 
  \frac{\lambda}{2}\int\abs{u - d}^2,\qquad
  E'[u] = -\nabla^2u + \lambda(u - d),
\end{gather*}
it is not exact for finite computations.

Instead, if trying to implement this with a minimizer that requires high-accurate
derivatives, make sure that the energy is computed in a way that exactly defines the
appropriate minimization problem such as in the following case:
\begin{gather*}
  E[u] = \int\frac{-u\nabla^2 u}{2} + \frac{\lambda}{2}\int\abs{u - d}^2.
\end{gather*}

While this is formally equivalent in the continuum limit, it is also exactly equivalent
numerically to the numerical implementation of $E'[u]$ as long as the matrix
representing $\mat{D}_2 \approx \nabla^2$ is symmetric $\mat{D}_s^T = \mat{D}_2$.

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
[machine precision]: <https://en.wikipedia.org/wiki/Machine_epsilon>
[Renormalization Group]: <https://physics-552-quantum-iii.readthedocs.io/en/latest/RenormalizationGroup.html>
[analytic function]: <https://en.wikipedia.org/wiki/Analytic_function>
[ringing artifacts]: <https://en.wikipedia.org/wiki/Ringing_artifacts>
