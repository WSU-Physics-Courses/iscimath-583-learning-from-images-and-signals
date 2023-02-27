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

import mmf_setup;mmf_setup.nbinit()
import logging;logging.getLogger('matplotlib').setLevel(logging.CRITICAL)
%matplotlib inline
import numpy as np, matplotlib.pyplot as plt
from IPython.display import clear_output, display
from myst_nb import glue
```

(sec:NonlocalMeans)=
# Non-local Means

The idea behind the non-local means algorithm is that random noise can be reduced by
averaging.  The challenge is to find pixels with random noise that can be averaged.  One
obvious solution is to take multiple pictures of the same object, then to average over
images.  Another approach is useful if the underlying image is smooth.  In this case,
one might obtain a reduction in noise by averaging nearby pixels, at the cost of
slightly bluring the image.

The idea of non-local means is to find regions of the image that are similar to use for
averaging, but to allow these regions to be spatially separated (non-local).  This
allows one to take advantage of repetition of features image such as repeated features
like windows, straight edges like streets or buildings.

:::{admonition} [Photographing a Black Hole][]

I think that this was the approach used to produce the image with the [Event Horizon Telescope (EHT)][EHT], but I have not been able to find an easy explanation or source of raw images.  This could be a fun example.

:::

+++

## Noise Reduction: Adding Gaussian Variables

To make this quantitative, we consider Gaussian noise $η \sim \mathcal{N}(0, σ^2)$ with
zero mean and standard deviation $σ$ with a [probability density function (PDF)][PDF]

\begin{gather*}
  p(η) = \frac{1}{\sqrt{2π σ^2}}\exp\left(\frac{-(η-μ)^2}{2σ^2}\right).
\end{gather*}

When we add Gaussian variables, we find:

\begin{gather*}
  η_1 \sim \mathcal{N}(μ_1, σ_1^2), \qquad
  η_2 \sim \mathcal{N}(μ_2, σ_2^2), \qquad
  aη_1 + bη_2 \sim \mathcal{N}\bigl(aμ_1 + bμ_2, (aσ_1)^2 + (bσ_2)^2\big).
\end{gather*}

*If you don't know this, you should derive it.  Here we check it numerically.*

```{code-cell} ipython3
%matplotlib inline
import numpy as np, matplotlib.pyplot as plt
import scipy.stats
sp = scipy

rng = np.random.default_rng(seed=2)
Ns = 100000
μ1, σ1 = 1, 0.2
μ2, σ2 = 2, 0.4
a, b = 1.2, 0.8
η1 = rng.normal(loc=μ1, scale=σ1, size=Ns)
η2 = rng.normal(loc=μ2, scale=σ2, size=Ns)
η12 = a*η1 + b*η2
μ12 = a*μ1 + b*μ2
σ12 = np.sqrt((a*σ1)**2 + (b*σ2)**2)

N1 = sp.stats.norm(loc=μ1, scale=σ1)
N2 = sp.stats.norm(loc=μ2, scale=σ2)
N12 = sp.stats.norm(loc=μ12, scale=σ12)

fig, ax = plt.subplots()
x = np.linspace(0, 4, 200)
args = dict(bins=100, alpha=0.5, density=True)
ax.hist(η1, label='$η_1$', **args)
ax.hist(η2, label='$η_2$', **args)
ax.hist(η12, label='$η_1+η_2$', **args)
ax.plot(x, N1.pdf(x), 'C0')
ax.plot(x, N2.pdf(x), 'C1')
ax.plot(x, N12.pdf(x), 'C2')
ax.set(xlabel="η")
ax.legend();
```

:::{margin}
Note that the reduction comes from the denominator: the sum has a large variance $Nσ^2$,
but dividing by $N$ reduces $σ \rightarrow σ/N$, resulting in the overall reduction in
the variance $σ^2\rightarrow σ^2/N$.
:::
Thus, if we take the average of $N$ samples $x=x_0 + η$, each with the same noise $η
\sim \mathcal{N}(0, σ^2)$, then

\begin{gather*}
  \bar{x} = \frac{x_1 + x_2 + \cdots + x_N}{N}, \qquad
  \bar{x} \sim \mathcal{N}(x_0, σ^2/N).
\end{gather*}

I.e., the standard deviation $σ \rightarrow σ/\sqrt{N}$ is thus reduced by a factor of
$\sqrt{N}$.  This is the essense of denoising by averaging. 

### Patches

The second aspect of non-linear means is that we must define a measure that describes
the **distance** between two pixels such that large values of this distance mean that
the pixels are not similar, but small distances mean that the pixels are similar and
suitable for averaging.

:::{margin}
Note: in terms of the data, we index as `u[ix, iy]` – so-called `'ij'` indexing.  This
corresponds to going down first, then across.  I.e. for a $2×3$ array
\begin{gather*}
  \mat{u} = \begin{pmatrix}
    0 & 3\\
    1 & 4\\
    2 & 5
  \end{pmatrix}.
\end{gather*}
By convention, when we display images, we display the *transpose* of this, so that $i_x$
goes *across* the image first:
\begin{gather*}
  \text{image} = \begin{bmatrix}
    0 & 1 & 2\\
    3 & 4 & 5
  \end{bmatrix}.
\end{gather*}
:::
We shall label the pixels in our image by an index $i \in \{0, 1, \dots , N_xN_y -1\}$
corresponding to the pixel $(i_x, i_y)$ in the image where
\begin{gather*}
  i = i_x + N_x i_y.
\end{gather*}
We will compute our distance between pixels by comparing **patches** of shape `(dx, dy)`
centered on the pixel.

One can play with many different distance functions $d(i, j)$: here we shall only
consider one, which is the average of the square of the differences between each patch,
optionally subtracting the mean (`subtract_mean=True` in the code).

If the patches match exactly except for the noise, then this corresponds to the average
of the square of $N=d_xd_y$ differences $\sum_{i=1}^{N} (η^{(A)}_i - η^{(B)}_i)^2 / N$ of
the noise $\eta^{(A,B)}_{i}$ in the $i$'th pixel of patch $A$ and $B$ respectively.  How are such
quantities distributed?  The answer can be expressed in terms of the famous [chi-squared
distribution][].

:::{margin}
The standard [chi-squared distribution][] has the following mean $k$ and variance $2k$.
:::
The [chi-squared distribution][] is sum of the squares of $N$ normally distributed
variables with unit variance $\sigma^2=1$:
\begin{gather*}
  \chi^2 = \sum_{i=1}^{N} η_i^2 \sim \chi^2(k=N), \qquad
  p(\chi^2; N) = \frac{(\chi^2)^{\frac{N}{2}-1}e^{-\chi^2/2}}{2^{N/2}\left(\frac{N}{2}-1\right)!}.
\end{gather*}
Here the parameter $N$ is referred to as the number of **degrees of freedom**.
Alternatively, this can be viewed as the radial distribution of an $N$-dimensional
[multivariate normal distribution][] with unit covariance $\mat{\Sigma} = \mat{1}$
(i.e. spherically symmetric):
:::{margin}
To get the power of $\chi^{N-2}$ and the factor of $2^{-N/2}$, recall that
\begin{gather*}
  \int \d{r}\;\delta\bigl(f(r)-f(x)\bigr)g(r)
  =
  \frac{g(x)}{f'(x)}\\
  =
  \int \frac{\d{f}}{\d{f}/\d{r}}\;\delta\bigl(f(r)-f(x)\bigr) g(r)\\
\end{gather*}
:::
\begin{gather*}
  p_{\mathcal{N}(\vect{0}, \mat{1})}(\vect{x})
  = \frac{1}{\sqrt{(2\pi)^{N}}}\exp\left(\frac{\norm{\vect{x}}_2^2}{2}\right), \\
  \begin{aligned}
    P(\chi^2;N) &= \braket{\delta(r^{2}-\chi^2)}_{\mathcal{N}(\vect{0}, \mat{1})} \\
    &= 
    \underbrace{\int\d{\Omega_{N}}}
              _{S_{N-1}=\frac{2\pi^{N/2}}{\Gamma(N/2)}}
    \int_0^{\infty}\d{r}\;
    \delta(r^2-\chi^2)r^{N-1}
    \frac{1}{\sqrt{(2\pi)^{N}}}e^{-r^2/2}\\
    &=
    \frac{\chi^{N-2}e^{-\chi^2/2}}
         {2^{N/2}\Gamma\bigl(\frac{N}{2}\bigr)}.
  \end{aligned}
\end{gather*}

If we subtract that mean, then we need to reduce this by one $p(\chi^2; N-1)$ (see [Cochran's
theorem][]).  Finally, these results are for $N$ normal variables with unit variance
$\sigma^2=1$.  Our patch differences are each the difference of 2 normal variables
$\eta^{(A)} - \eta^{(B)}$ and then we use the mean which includes a factor of $1/N$.
The resulting distribution is codified in {data}`scipy.stats.chi2` if we include an
additional scaling factor of $\texttt{scale=}2σ^2/N$ and set the correct number of
degrees of freedom to $\texttt{df}=N$ or $\texttt{df}=N-1$ depending on the value of
`subtract_mean`.

```{code-cell} ipython3
dx, dy = 2, 2
N = dx*dy
σ = 0.2

A, B = rng.normal(scale=σ, size=(2, dx*dy, Ns))
dAB = A - B
d0 = (dAB**2).mean(axis=0)                     # subtract_mean = False
d1 = ((dAB-dAB.mean(axis=0))**2).mean(axis=0)  # subtract_mean = True

chi2_0 = sp.stats.chi2(scale=2*σ**2/N, df=N)
chi2_1 = sp.stats.chi2(scale=2*σ**2/N, df=N-1)
fig, ax = plt.subplots()
ax.hist(d0, label="subtract_mean=False", **args);
ax.hist(d1, label="subtract_mean=True", **args);
x = np.linspace(-0, 0.3, 200)
ax.plot(x, chi2_0.pdf(x), 'C0')
ax.plot(x, chi2_1.pdf(x), 'C1')
ax.set(xlim=(-0.01, 0.3), xlabel="$d(A, B)$")
ax.legend();
```

Thus, if we know $σ$, and we have perfectly matching patches in our image, we can use
the inverse cumulative distribution to determine the **threshold** below which we should
average.  This is complicated in the real world where "matching" patches will still have a
non-zero distance – even in the absence of noise.

An alternative is to request a certain level of noise reduction $s$, which means that we
should average at least $s^2$ pixels.  Here we can sort the data and choose the minimum
number.  We provide both options in the code through the argument `Nsamples` of
{func}`denoise.NonLocalMeans.get_threshold`.

The situation is more complicated if the patches do not match.  We currently do not have
a good model or theory for how to deal with this.  Instead, we take a phenomenological
approach of performing a weighted average with weights.  The default implementation is
\begin{gather*}
  f(d) = e^{-d^2/2\sigma^2}
\end{gather*}
but $f=$`f_weight` can be specified by the user.  Here
$\sigma=$`sigma_weight`$(\vect{d})$ is a function of the weights $\vect{d}$ in the 
"neighbourhood" of the pixel in consideration as defined by the specified threshold or
`Nsamples`.  The default implementation is
\begin{gather*}
  \sigma = \frac{\max\vect{d}}{2\texttt{k_sigma}}.
\end{gather*}

:::{admonition} To Do (Advanced)!

Can you come up with a rigorous model for which a specific choice of weighting function
$f(d)$ is optimal in some sense?
:::

## Demonstration

To demonstrate the non-local means algorithm, we start with an example that should work
well.  Here we define a repeating $5\times 6$ grid with an overall trend.  The idea is
that we can average over each of the $5\times 6=30$ repeating units to reduce the
variance of the noise by factor of $30$.

```{code-cell} ipython3
import scipy.stats
from math_583 import denoise

sp = scipy

class TestImage(denoise.Image):
    dx, dy = 5, 4      # Patch size
    Npx, Npy = 3, 5    # Number of patches
    mx, my = 1.0, 2.0  # Slopes of trend
    
    def init(self):
        dx, dy = self.dx, self.dy
        Npx, Npy = self.Npx, self.Npy
        Nx, Ny = Npx*dx, Npy*dy
        x, y = np.ogrid[:Nx, :Ny]
        u_exact = self.mx * x / Nx + self.my * y / Ny + (x%dx == 0) + (y%dy == 0)
        self._data = u_exact
        super().init()

    def denoise(self, u, subtract_mean=True):
        """Return an "optimally" denoised image using the patch size."""
        dx, dy = self.dx, self.dy
        Npx, Npy = self.Npx, self.Npy
        U = u.reshape((Npx, dx, Npy, dy))
        U_ = np.zeros([Npx, 1, Npy, 1])
        if subtract_mean:
            U_ = U.mean(axis=1).mean(axis=2)[:, None, :, None]
        u_denoise = ((U - U_).mean(axis=0).mean(axis=1)[None, :, None, :] 
                     + U_).reshape(u.shape)
        return u_denoise
    
rng = np.random.default_rng(seed=2)

im = TestImage()
u_exact = im.get_data()

sigma = 0.2
u_noise = u_exact + rng.normal(scale=sigma, size=u_exact.shape)
u0 = im.denoise(u_noise)
glue('Npatch', im.Npx*im.Npy, display=False)
im.show(u0, u_noise, u_exact)
```

Here, the denoised image `u` is obtained by averaging over the {glue:}`Npatch` patches.
Since the image has an overall trend, we first subtract the mean of each patch in `U_`,
then we average over all the resulting {glue:}`Npatch` patches.  Finally we restore the
mean for each patch.  The final image `u` thus consists of {glue:}`Npatch` patches, each
with the original mean, but with the same fluctuations about this mean.

## Quality of Fit Metric 

To quantify the degree of success of our fitting, we define the quality of fit as the
average square deviation of the pixels:
\begin{gather*}
  d(\mat{u}, \mat{u}_\text{exact}) = \sum_{i_x, i_y} 
  \frac{[\mat{u} - \mat{u}_{\text{exact}}]_{ij}^2}{N_xN_y}
  = \frac{\norm{\mat{u} - \mat{u}_{\text{exact}}}^2_{2}}{N_x,N_y}.
\end{gather*}

For a noisy image with Gaussian noise of variance $\sigma^2$, the distance will be
distributed as
\begin{gather*}
  d(\mat{u}, \mat{u}_\text{exact}) \sim \frac{\sigma^2}{N_xN_y} \chi^2(k=N_xN_y).
\end{gather*}
After averaging over the {glue:}`Npatch` patches, we expect that the variance should
down by this factor $\sigma^2 \rightarrow \sigma^2 / ${glue:}`Npatch`, and this is the
case if we have `subtract_mean=False`.  However, the situation with `subtract_mean=True`
is more complicated.

:::{admonition} To Do (Advanced).

Can you figure out how the distances are distributed in the case of `subtract_mean=True`
or with a linear trend in the images?
:::

```{code-cell} ipython3
rng = np.random.default_rng(seed=2)
im = TestImage()
im_no_trend = TestImage(mx=0, my=0)

Ns = 10000

# Keys are no_trend or (no_trend, subtract_mean)

u_exacts = {
    False: im.get_data(),
    True: im_no_trend.get_data(),
}

noise_dists = {
  no_trend: []
  for no_trend in [True, False]
}

denoise_dists = {
  (no_trend, subtract_mean): []
  for no_trend in [True, False]
  for subtract_mean in [True, False]
}

sigma = 0.4

for n in range(Ns):
    for no_trend in [True, False]:
        u_exact = u_exacts[no_trend]
        u_noise = u_exact + rng.normal(scale=sigma, size=im.shape)
        noise_dists[no_trend].append(((u_noise - u_exact)**2).mean())
        for subtract_mean in [True, False]:
            key = (no_trend, subtract_mean)
            u0 = im.denoise(u_noise, subtract_mean=subtract_mean)
            denoise_dists[key].append(((u0 - u_exact)**2).mean())

args = dict(bins=100, alpha=0.5, density=True)

fig, ax = plt.subplots()
for dist, c, label in [
        (noise_dists[True], 'C0', 'noise (trend and no_trend)'),
        (noise_dists[False], 'C0', ''),
        (denoise_dists[(True, False)], 'C2', 'denoise: no_trend'),
        (denoise_dists[(True, True)], 'C4', 'denoise: subtract_mu, (trend and no_trend)'),
        (denoise_dists[(False, True)], 'C4', ''),
        (denoise_dists[(False, False)], 'red', 'denoise: trend')]:
    k = 2*(np.mean(dist)/np.std(dist))**2
    ax.hist(dist, color=c, label=label, **args)

# We can calculate the distribution in 2 cases:
# Pure noise: the exact distribution should be a chi^2 distribution from 
# averaging the square deviation over all df=N pixels.  The averaging needs
# a scaling sigma**2/N
df = N = np.prod(im.shape)
chi2 = sp.stats.chi2(scale=sigma**2/N, df=df)
x = np.linspace(0, max(noise_dists[True]), 1000)
ax.semilogx(x, chi2.pdf(x), c="C0")

# No trend and no subtract_mu.  Here we will gain a reduction in noise by 
# factor = Np, the number of patches.  The final image will be composed of
# Np identical copies of the average patch, which has df = dx*dy degrees of
# freedome.  Thus, we will have 
factor = Np = im.Npx * im.Npy
# No trend and no subtract_mean means we have df = dx*dy independent degrees of freedom
# since each patch is repeated multiple times.
df = im.dx*im.dy
chi2_denoise = sp.stats.chi2(scale=Np * sigma**2 / N / factor, df=df)
ax.semilogx(x, chi2_denoise.pdf(x), c="C2")

# Can you figure out the distributions with a trend, or with subtract_mu == True?
ax.legend()
ax.set(xlabel="dist(u, u_exact)", ylim=(0, 200), xlim=(0.001, 0.3));
```

## Non-Local Means

We can now apply our implementation of the non-local means algorithm to see how well it
performs.

```{code-cell} ipython3
import scipy.stats
from importlib import reload
from math_583 import denoise; reload(denoise)

sp = scipy

rng = np.random.default_rng(seed=2)

sigma = 0.2
patch_size = 5
im = TestImage()
u_exact = im.get_data()
u_noise = u_exact + rng.normal(scale=sigma, size=im.shape)
u0 = im.denoise(u_noise, subtract_mean=True)
nlm = denoise.NonLocalMeans(im, dx=patch_size, dy=patch_size,
                            mode='reflect', 
                            sigma=sigma, subtract_mean=True, symmetric=True)
im.show(u0, u_noise, u_exact)
err = nlm.dist(u0, u_exact)
plt.suptitle(f"{err=:.2g}")

# These can be precomputed for speed
u = nlm.denoise(u_noise, k_sigma=0, percentile='optimize', u_exact=u_exact)
im.show(u, u_noise, u_exact)
err = nlm.dist(u, u_exact)
plt.suptitle(f"{err=:.2g}");
```

### Scikit Image
Here is a comparison with the implementation in
[scikit-image](https://scikit-image.org/docs/dev/api/skimage.restoration.html#denoise-nl-means):

```{code-cell} ipython3
from scipy.optimize import minimize_scalar
from skimage.restoration import denoise_nl_means, estimate_sigma
kw = dict(patch_size=patch_size, patch_distance=5, fast_mode=False, sigma=sigma)
_cache = {}
def err(h_sigma):
    if not h_sigma in _cache:
        u = denoise_nl_means(u_noise, h=h_sigma*sigma, **kw);
        _cache[h_sigma] = nlm.dist(u, u_exact)
    return _cache[h_sigma]
res = minimize_scalar(err, bracket=(0.1, 0.5))
print(f"optimal_h_sigma={res.x:.4g}")
h_sigma = res.x
u = denoise_nl_means(u_noise, h=h_sigma*sigma, **kw)
im.show(u, u_noise, u_exact)
err = nlm.dist(u, u_exact)
plt.suptitle(f"{err=:.2g}");
```

<!--
```{code-cell} ipython3
n = denoise.NonLocalMeans(im, mode='reflect', sigma=sigma, subtract_mean=True, symmetric=True)
u_ = n.pad(u_noise)
dists = n.compute_dists(u_=u_)
```

```{code-cell} ipython3
u = n.denoise(u_noise, dists=dists, u_=u_, percentile=75.0)
im.show(u, u_noise, u_exact)
```

```{code-cell} ipython3
from tqdm.auto import tqdm
ps = 100 - 10**np.linspace(-1, 1.9, 20)
#ps = 100 - 10**np.linspace(-7, 0, 20)
errs = [n.dist(u_exact, n.denoise(u_noise, dists=dists, u_=u_, percentile=_p))
        for _p in tqdm(ps)]
#im.show(u, u_noise, u_exact)
```

```{code-cell} ipython3
fig, ax = plt.subplots()
ax.plot(ps, errs)
ax.axhline(n.dist(u0, u_exact), c='y')
```

```{code-cell} ipython3
u = n.denoise(u_noise, dists=dists, u_=u_, percentile=80)
im.show(u, u_noise, u_exact)
```







```{code-cell} ipython3
def i_(ix, iy):
    """Return the linear patch index."""
    return ix + (Nx-dx)*iy

def ixy_(i):
    """Return (ix, iy) from the linear patch index i."""
    nx = Nx - dx
    iy = i // nx
    ix = i % nx
    return ix, iy

def dist(A, B):
    dAB = A - B
    dAB_ = dAB.mean()
    return ((dAB - dAB_)**2).mean()


im.show(u0, u_noise, u_exact)

ix, iy = ixy_(100)
plt.plot([ix*dx, ix*dx, ix*dx+dx, ix*dx+dx, ix*dx], 
         [iy*dy, iy*dy+dy, iy*dy+dy, iy*dy, iy*dy], '-')
```

```{code-cell} ipython3
def get_patch(u, i):
    ix, iy = ixy_(i)
    return u[ix:ix+dx, iy:iy+dy]

def compute_dists(u, dx=dx, dy=dy):
    Nx, Ny = u.shape
    nx, ny = Nx - dx, Ny - dy
    Np = nx*ny  # Number of patches
    dists = np.array([
        [0] * (i0+1)
        + [dist(get_patch(u, i0), get_patch(u, i1)) 
           for i1 in range(i0+1, Np)]
        for i0 in range(Np)])
    dists += dists.T
    return np.ma.masked_array(dists, mask=np.eye(Np))
```

```{code-cell} ipython3
im.show(D, vmin=0, vmax=D.max())
```

```{code-cell} ipython3
D = compute_dists(u_noise)
plt.hist(D.ravel(), 40, density=True);
x = np.linspace(D.min(), D.max(), 200)

# Factor of 2 because D is symmetric
chi2 = sp.stats.chi2(scale=2*sigma**2/dx/dy, df=dx*dy)
plt.plot(x, chi2.pdf(x))
```

The $\chi^2_N$ distribution with $N$ degrees of freedom (`df`) is distribution of the sum of the squares of $f$ normally distributed variables.  Since this is the square, the `scale` is $\sigma^2$.  In our case, we take the mean, so we introduce an additional factor of $N$ in the scale.

```{code-cell} ipython3
x = np.linspace(0,1,100)
sigma = 0.5
N = 25
samples = (rng.normal(scale=sigma, size=(10000, N))**2).sum(axis=1)
plt.hist(samples, 
         40,
         density=True);
x = np.linspace(0, samples.max(), 200)
plt.plot(x, sp.stats.chi2(scale=sigma**2, df=N).pdf(x))
```

```{code-cell} ipython3
x = np.linspace(0,1,100)
sigma = 0.1
N = 25
samples = (rng.normal(scale=sigma, size=(10000, N))**2).mean(axis=1)
plt.hist(samples, 
         40,
         density=True);
x = np.linspace(0, samples.max(), 200)
plt.plot(x, sp.stats.chi2(scale=sigma**2/N, df=N).pdf(x))
```

```{code-cell} ipython3
threshold = chi2.ppf(0.5)
threshold = 0.01
```

```{code-cell} ipython3
from IPython.display import clear_output
import time
Np = D.shape[0]
ds = sorted([
    (D[i0, i1], (i0, i1))
    for i0 in range(Np-1) 
    for i1 in range(i0+1, Np)])

u0 = u_noise.copy()
u = u_noise.copy()

for (d, (i0, i1)) in ds:
    if d > threshold:
        break
    #w = np.exp(-(d/threshold))
    w = 1
    ixy0, ixy1 = ixy_(i0), ixy_(i1)
    p0 = u[ixy0[0]:ixy0[0]+dx, ixy0[1]:ixy0[1]+dy]
    p1 = u[ixy1[0]:ixy1[0]+dx, ixy1[1]:ixy1[1]+dy]
    p0_ = (p0+w*(p1-p1.mean()+p0.mean()))/(1+w)
    p1_ = (p1+w*(p0-p0.mean()+p1.mean()))/(1+w)
    #p0_ = (p0+w*p1)/(1+w)
    #p1_ = (p1+w*p0)/(1+w)
    u[ixy0[0]:ixy0[0]+dx, ixy0[1]:ixy0[1]+dy] = p0_
    u[ixy1[0]:ixy1[0]+dx, ixy1[1]:ixy1[1]+dy] = p1_
    if False:
        im.show(u, u_noise, u_exact)
        for (ix, iy) in [ixy0, ixy1]:
            plt.plot([iy, iy+dy, iy+dy, iy, iy], 
                     [ix, ix, ix+dx, ix+dx, ix], '-')
        display(plt.gcf())
        clear_output(wait=True)
        #time.sleep(0.1)
im.show(u, u_noise, u_exact)
```

## Development Notes

+++

Here I document some of the ideas and issues I faced when developing the non-local means algorithm.

Originally I wanted to index each pixel and then compute the neighbourhood as ±`dx`, ±`dy` pixels in each direction.  This leads to some complicated indexing like `u[ix-dx:ix+dx+1,iy-dy:iy+dy+1]`.  This is kind of a pain, so I opted to change this and instead index each "patch" simply by the upper left pixel, reinterpreting `dx` and `dy` as the full width and height of the patches: `u[ix:ix+dx, iy:iy+dy]`.

The second issue is that of boundary conditions.  Currently I am opting to define these by appropriately padding the original array.  This is the task of `NonLocalMeans.pad()`.

An alternative might be to use a more graph-based method, and then define the apppropriate neighbours through adjacency... maybe later.

+++

### Padding

Padding is a little tricky.  We test the indexing here.

```{code-cell} ipython3
# Design this to work with odd dx, tweak to work with even dx.
import numpy as np
Nx = 6           # Size of original array
dx = 3           # Size of patch
Nx_ = Nx + dx-1  # Size of padded array
ix0 = dx//2      # Left index of original array in padded array
ix1 = (dx-1)//2  # Right index of original array in padded array

u = 1+np.arange(Nx)
u_ = np.zeros_like(u, shape=(Nx_))
u_[ix0:ix0+Nx] = u
print(u_)

u_[:ix0] = u_[Nx:][:ix0]   # Easy way to get periodic boundary condition
u_[-ix1:] = u_[ix0:][:ix1] # Tweak to work with 
print(u_)
```

```{code-cell} ipython3
import mmf_setup;mmf_setup.set_path();
from importlib import reload
from math_583 import denoise;reload(denoise)
u = np.array([[1, 2, 3], 
              [4, 5, 6]])
im = denoise.Image(u)
n = denoise.NonLocalMeans(im, mode='wrap')
n.pad(u)
```

```{code-cell} ipython3
A, B = np.random.random((2, 5, 5))
%timeit np.mean(A)
%timeit A.mean()
```

Here is how we index into an array.  Note that the y-index should increase inside.

```{code-cell} ipython3
print(np.ravel([[1, 2, 3], 
                [4, 5, 6]]))
```

```{code-cell} ipython3
n = denoise.NonLocalMeans(im, mode='wrap')
n.compute_dists(u).reshape(6, 6)
```

```{code-cell} ipython3
import mmf_setup;mmf_setup.set_path();
from importlib import reload
from math_583 import denoise;reload(denoise)
u = np.array([[1, 2, 3], 
              [4, 5, 6]])
n = denoise.NonLocalMeans(denoise.Image(u), mode='wrap')
n.compute_dists(u).reshape(6, 6)
```

### Our Code

```{code-cell} ipython3
%pylab inline
import mmf_setup;mmf_setup.nbinit()
import numpy as np, matplotlib.pyplot as plt
import scipy.stats
from importlib import reload
from math_583 import denoise; reload(denoise)

sp = scipy

rng = np.random.default_rng(seed=2)

dx, dy = (5, 5)
Npx, Npy = (5, 6)

Nx, Ny = Npx*dx, Npy*dy

x, y = np.ogrid[:Nx, :Ny]
u = 0.0 + (x%dx == 0) + (y%dy == 0)
u = x/Nx + 2*y/Ny + (x%dx == 0) + (y%dy == 0)
#u = 0*x*y + 0.5

sigma = 0.1
im = denoise.Image(u)
u_exact = im.get_data()
u_noise = u_exact + rng.normal(scale=sigma, size=(Nx, Ny))

U = u_noise.reshape((Npx, dx, Npy, dy))
U_ = U.mean(axis=1).mean(axis=2)[:, None, :, None]
u0 = ((U - U_).mean(axis=0).mean(axis=1)[None, :, None, :] + U_).reshape(u_noise.shape)
im.show(u0, u_noise, u_exact)
n = denoise.NonLocalMeans(im, mode='reflect', sigma=sigma, subtract_mean=True)
```

```{code-cell} ipython3
n = denoise.NonLocalMeans(im, mode='reflect', sigma=sigma, subtract_mean=True, symmetric=True)
u_ = n.pad(u_noise)
dists = n.compute_dists(u_=u_)
```

```{code-cell} ipython3
u = n.denoise(u_noise, dists=dists, u_=u_, percentile=75.0)
im.show(u, u_noise, u_exact)
```

```{code-cell} ipython3
from tqdm.auto import tqdm
ps = 100 - 10**np.linspace(-1, 1.9, 20)
#ps = 100 - 10**np.linspace(-7, 0, 20)
errs = [n.dist(u_exact, n.denoise(u_noise, dists=dists, u_=u_, percentile=_p))
        for _p in tqdm(ps)]
#im.show(u, u_noise, u_exact)
```

```{code-cell} ipython3
fig, ax = plt.subplots()
ax.plot(ps, errs)
ax.axhline(n.dist(u0, u_exact), c='y')
```

```{code-cell} ipython3
u = n.denoise(u_noise, dists=dists, u_=u_, percentile=80)
im.show(u, u_noise, u_exact)
```

```{code-cell} ipython3
rng = np.random.default_rng(seed=2)

dx, dy = (5, 5)
Npx, Npy = (5, 6)

Nx, Ny = Npx*dx, Npy*dy

x, y = np.ogrid[:Nx, :Ny]
u = x/Nx + 2*y/Ny + (x%dx == 0) + (y%dy == 0)

im = denoise.Image(u)

def get_dist(log10p100s=(0.1,), sigma=0.1, dx=5, dy=5, subtract_mean=True, mode='reflect', im=im):
    """Return (percentiles, [dist(u, u_exact)]).
    
    Parameters
    ----------
    sigma : float
        Std of noise.
    log10p1002 : [float]
        List of log10(100 - percentile)
    """
    Nx, Ny = im.shape
    u_exact = im.get_data()
    u_noise = u_exact + rng.normal(scale=sigma, size=(Nx, Ny))
    n = denoise.NonLocalMeans(im, mode=mode, sigma=sigma, subtract_mean=subtract_mean, dx=dx, dy=dy)
    u_ = n.pad(u_noise)
    dists = n.compute_dists(u_=u_)
    percentiles = 100 - 10**np.asarray(log10p100s)
    errs = [n.dist(u_exact, n.denoise(u_noise, percentile=_p, dists=dists, u_=u_))
            for _p in tqdm(percentiles)]
    return percentiles, errs
```

```{code-cell} ipython3
log10p100s = np.linspace(-1.5, 1.5, 20)
ds = [3, 5, 7]
ps_errs = [get_dist(log10p100s, dx=_d, dy=_d) for _d in ds]
```

```{code-cell} ipython3
fig, ax = plt.subplots()
for d, (ps, errs) in zip(ds, ps_errs):
    ax.plot(ps, errs, label=f"dx=dy={d}")
ax.legend()
ax.set(xlabel="threshold percentile", ylabel="dist(u, u_exact)")
```

```{code-cell} ipython3
Nx, Ny = u_noise.shape
Np = Nx*Ny
D = np.ma.masked_array(D, mask=np.eye(Np))
im.show(D, vmin=0, vmax=D.max())
```

```{code-cell} ipython3
ds = sorted([(D[i0, i1], (i0, i1)) for i1 in range(Np) for i0 in range(i1+1,Np)])
plt.plot([_d[0] for _d in ds])
plt.axhline(n.get_threshold(), c='y')
```

```{code-cell} ipython3
threshold = n.get_threshold()
neighbours = {}
for _d, (i0, i1) in ds:
    if _d > threshold:
        break
    neighbours.setdefault(i0, []).append((i1, _d))
    neighbours.setdefault(i1, []).append((i0, _d))

Nx, Ny = u_noise.shape
u = np.zeros((Nx, Ny))
for ix in range(Nx):
    for iy in range(Ny):
        i0 = n.i(ix, iy)
        u[ix, iy] = np.mean([u_noise[ix, iy]] + [u_noise[n.ixy(i1)] for (i1, d) in neighbours.get(i0, [(i0, 1)])])
        
```

```{code-cell} ipython3
im.show(u, u_noise, u_exact)
```

```{code-cell} ipython3
[u_noise[n.ixy(i1)] for (i1, d) in neighbours.get(i0, [(i0, 1)])]
```

```{code-cell} ipython3
ds0 = [_d for _d in ds if _d[0] <= threshold]
im.show(u_noise)
for (_d, (i0, i1)) in ds0:
    ixy0 = n.ixy(i0)
    ixy1 = n.ixy(i1)
    plt.plot([ixy0[1], ixy1[1]], [ixy0[0], ixy1[0]], '-x', lw=0.1)
```

```{code-cell} ipython3

```

```{code-cell} ipython3
im.show(u_noise)
plt.plot([0], [1], 'x')
```

```{code-cell} ipython3
plt.hist(D.ravel(), 40, density=True);
plt.axvline(n.get_threshold(), c='y')
```

```{code-cell} ipython3
u_ = n.pad(u_noise)
im.show(u_, vmin=0, vmax=u_.max())
```

```{code-cell} ipython3
def dist(A, B):
    dAB = A - B
    dAB_ = dAB.mean()
    return ((dAB - dAB_)**2).mean()


dx, dy = n.dx, n.dy
i0 = 0
i1 = 1
(ix0, iy0), (ix1, iy1) = n.ixy(i0), n.ixy(i1)
A = u_[ix0:ix0+dx, iy0:iy0+dy] 
B = u_[ix1:ix1+dx, iy1:iy1+dy]
n.dist(A, B), D[i0, i1], D_[ix0, iy0, ix1, iy1], D_[ix1, iy1, ix0, iy0]
```

```{code-cell} ipython3
ix1, iy1
```

```{code-cell} ipython3
n = denoise.NonLocalMeans(im, mode='wrap')
u_ = n.pad(u)
im.show(u_, vmin=0, vmax=u_.max())
```

```{code-cell} ipython3

```

-->

[PDF]: <https://en.wikipedia.org/wiki/Probability_density_function>
[EHT]: <https://eventhorizontelescope.org/>
[Photographing a Black Hole]: <https://www.nasa.gov/image-feature/photographing-a-black-hole>
[Cochran's theorem]: <https://en.wikipedia.org/wiki/Chi-squared_distribution#Cochran's_theorem>
[chi-squared distribution]: <https://en.wikipedia.org/wiki/Chi-squared_distribution>
[multivariate normal distribution]: <https://en.wikipedia.org/wiki/Multivariate_normal_distribution>
