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
```

(sec:NonlocalMeans)=
# Non-local Means

+++

The idea behind the non-local means algorithm is that random noise can be reduced by averaging.  The challenge is to find pixels with random noise that can be averaged.  One obvious solution is to take multiple pictures of the same object, then to average over images.  Another approach is useful if the underlying image is smooth.  In this case, one might obtain a reduction in noise by averaging nearby pixels, at the cost of slightly bluring the image.

The idea of non-local means is to find regions of the image that are similar to use for averaging, but to allow these regions to 

:::{admonition} [Photographing a Black Hole][]

I think that this was the approach used to produce the image with the [Event Horizon Telescope (EHT)][EHT], but I have not been able to find an easy explanation or source of raw images.  This could be a fun example.

:::

[EHT]: <https://eventhorizontelescope.org/>
[Photographing a Black Hole]: <https://www.nasa.gov/image-feature/photographing-a-black-hole>

+++

## Noise and Probabilies

+++

To make this quantitative, we consider Gaussian noise $η \sim \mathcal{N}(0, σ^2)$ with zero mean and standard deviation $σ$ with a [probability density function (PDF)][PDF]

\begin{gather*}
  p(η) = \frac{1}{\sqrt{2π σ^2}}\exp\left(\frac{-x^2}{2σ^2}\right).
\end{gather*}

When we add Gaussian variables, we find:

\begin{gather*}
  η_1 \sim \mathcal{N}(μ_1, σ_1^2), \qquad
  η_2 \sim \mathcal{N}(μ_2, σ_2^2), \qquad
  aη_1 + bη_2 \sim \mathcal{N}(aμ_1 + bμ_2, (aσ_1)^2 + (bσ_2)^2).
\end{gather*}

*If you don't know this, you should derive it.  Here we check it numerically.*

[PDF]: <https://en.wikipedia.org/wiki/Probability_density_function>

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
ax.legend();
```

Thus, if we take the average of $N$ samples $x=x_0 + η$, each with the same noise $η \sim \mathcal{N}(0, σ^2)$, then

\begin{gather*}
  \bar{x} = \frac{x_1 + x_2 + \cdots + x_N}{N}, \qquad
  \bar{x} \sim \mathcal{N}(x_0, σ^2/N).
\end{gather*}

I.e., the standard deviation $σ \rightarrow σ/\sqrt{N}$ is thus reduced by a factor of $\sqrt{N}$.  This is the essense of denoising by averaging.

+++

The second aspect of non-linear means is that we must define a measure that describes the **distance** between two pixels such that large values of this distance mean that the pixels are not similar, but small distances mean that the pixels are similar and suitable for averaging.

:::{margin}
Note: in terms of the data, we index as `u[ix, iy]` – so-called `'ij'` indexing.  This corresponds to going down first, then across.  I.e. for a $2×3$ array
\begin{gather*}
  \mat{u} = \begin{pmatrix}
    0 & 3\\
    1 & 4\\
    2 & 5
  \end{pmatrix}.
\end{gather*}
By convention, when we display images, we display the *transpose* of this, so that $i_x$ goes *across* the image first:
\begin{gather*}
  \text{image} = \begin{bmatrix}
    0 & 1 & 2\\
    3 & 4 & 5
  \end{bmatrix}.
\end{gather*}
:::
We shall label the pixels in our image by an index $i \in \{0, 1, \dots , N_xN_y -1\}$ corresponding to the pixel $(i_x, i_y)$ in the image where
\begin{gather*}
  i = i_x + N_x i_y.
\end{gather*}
We will compute our distance between pixels by comparing **patches** of shape `(dx, dy)` centered on the pixel.

One can play with many different distance functions $d(i, j)$: here we shall only consider one, which is the average of the square of the differences between each patch, optionally subtracting the mean (`subtract_mean=True` in the code).

If the patches match exactly except for the noise, then this corresponds to the average of the square of $N=d_xd_y$ differences $\sum (η_i - η_j)^2 / N$.  How are such quantities distributed?  The sum of the squares of $N$ normally distributed variables is the famous [chi-squared distribution][]:
\begin{gather*}
  \chi^2 = \sum_{i=1}^{N} η_i^2, \qquad
  p(\chi^2; N) = \frac{(\chi^2)^{\frac{N}{2}-1}e^{-\chi^2/2}}{2^{N/2}\left(\frac{N}{2}-1\right)!}.
\end{gather*}
Here the parameter $N$ is referred to as the number of **degrees of freedom**.  If we subtract that mean, then we need to reduce this by one $p(\chi^2; N-1)$ (see [Cochran's theorem][]).  The various properties of this distribution are codified in {data}`scipy.stats.chi2`.  In our case, we need to adjust the scale by a factor of $2σ^2/N$ and set the correct number of degrees of freedom $N$ or $N-1$ depending on `subtract_mean`.


[Cochran's theorem]: <https://en.wikipedia.org/wiki/Chi-squared_distribution#Cochran's_theorem>
[chi-squared distribution]: <https://en.wikipedia.org/wiki/Chi-squared_distribution>

```{code-cell} ipython3
dx, dy = 2, 2
N = dx*dy
σ = 0.2

A, B = rng.normal(scale=σ, size=(2, dx*dy, Ns))
dAB = A - B
d0 = (dAB**2).mean(axis=0)
d1 = ((dAB-dAB.mean(axis=0))**2).mean(axis=0)

chi2_0 = sp.stats.chi2(scale=2*σ**2/N, df=N)
chi2_1 = sp.stats.chi2(scale=2*σ**2/N, df=N-1)
fig, ax = plt.subplots()
ax.hist(d0, label="subtract_mean=False", **args);
ax.hist(d1, label="subtract_mean=True", **args);
x = np.linspace(-0, 0.3, 200)
ax.plot(x, chi2_0.pdf(x), 'C0')
ax.plot(x, chi2_1.pdf(x), 'C1')
ax.set(xlim=(-0.01, 0.3))
ax.legend();
```

Thus, if we know $σ$, and we have perfectly matching patches in our image, we can use the inverse cumulative distribution to determine the **threshold** below which we should average.  This is complicated in the real world where even patches that correspond to a match will have a non-zero distance – even in the abscence of noise.

An alternative is to request a certain level of noise reduction $s$, which means that we should average at least $s^2$ pixels.  Here we can sort the data and choose the minimum number.  We provide both options in the code.

```{code-cell} ipython3
import numpy as np
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

sigma = 0.04
im = denoise.Image(u)
u_exact = im.get_data()
u_noise = u_exact + rng.normal(scale=sigma, size=(Nx, Ny))
u0 = np.asarray(np.bmat(
    [[
        u_noise.reshape((Npx, dx, Npy, dy)).mean(axis=0).mean(axis=1)
    ]*Npy]*Npx))
im.show(u0, u_noise, u_exact)
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
u0 = np.asarray(np.bmat(
    [[
        u_noise.reshape((Npx, dx, Npy, dy)).mean(axis=0).mean(axis=1)
    ]*Npy]*Npx))
im.show(u0, u_noise, u_exact)
n = denoise.NonLocalMeans(im, mode='wrap', sigma=sigma, subtract_mean=False)
```

```{code-cell} ipython3
n = denoise.NonLocalMeans(im, mode='wrap', sigma=sigma, subtract_mean=True)
u_ = n.pad(u_noise)
dists = n.compute_dists(u_=u_)
```

```{code-cell} ipython3
u = n.denoise(u_noise, dists=dists, u_=u_, percentile=95)
im.show(u, u_noise, u_exact)
```

```{code-cell} ipython3
D = n.compute_dists(u=u_noise)
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
