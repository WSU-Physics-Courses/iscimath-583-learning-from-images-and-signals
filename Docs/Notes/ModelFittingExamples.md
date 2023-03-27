---
execution:
  timeout: 300
jupytext:
  notebook_metadata_filter: all
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.0
kernelspec:
  display_name: Python 3 (math-583)
  language: python
  name: math-583
---

```{code-cell} ipython3
:tags: [hide-cell]

import mmf_setup

mmf_setup.nbinit()
import logging

logging.getLogger("matplotlib").setLevel(logging.CRITICAL)
%matplotlib inline
import numpy as np, matplotlib.pyplot as plt
```

(sec:ModelFittingEg)=
# Model Fitting Examples

Here we provide concrete examples following {ref}`sec:ModelFitting`

## Background
Before we work through some simple examples, we discuss a few analysis tools.

### Histograms

To get a feel for these distributions, we plot them with a histogram using
{py:func}`matplotlib.pyplot.hist`.

:::{admonition} Do It!
Use {py:func}`matplotlib.pyplot.hist` to make a histogram of $p(\eta)$.  Play with the
number of points and try to make it look good.  Use {data}`scipy.stats.norm` or a custom
[PDF][] to compare.
:::

```{code-cell}
import scipy.stats
sp = scipy

# Always seed your random numbers so you can reproduce your results
rng = np.random.default_rng(seed=2)

# Plot a normal distribution with Ns samples for a random sigma = (0, 10).
Ns = 1000
sigma = 10 * rng.random()
e = rng.normal(scale=sigma, size=Ns)

xlim = np.percentile(e/sigma, [0.1, 99.9])
hist_kw = dict(bins=40, density=True, histtype='step')

x = np.linspace(*xlim)
fig, ax = plt.subplots(figsize=(7, 2))
hist = ax.hist(e/sigma, **hist_kw)
ax.plot(x, sp.stats.norm().pdf(x))
ax.set(xlim=xlim, xlabel="$e/\sigma$");
```

Here are some tips to make nice-looking histograms:
1. If you have natural bins, use them.  We will see this later.  Here we do not, but
   we choose a reasonably large number of bins.  If you use too few bins, your histogram
   will be "blocky", but if you use too few, then you will just get a bunch of peaks.
2. If your distribution is wide, you might make a nicer plot if you limit the `xlim` to
   exclude the outliers, but don't use `range=xlim` in {py:func}`matplotlib.pyplot.hist`
   if you want to use `density=True`, because this will exclude outliers from the
   normalization.  {func}`numpy.percentile` is a great tool for this.
3. Set `density=True` if you want to compare with a [PDF][].
4. If you need to plot a bunch of data, `histtype='step'` will makes it easier to
   compare.
5. Note that you can use LaTeX for math `"$e/\sigma$"` in labels, or you can use unicode
   `"e/σ".  For publications you should do the former so that you can ultimately match
   fonts, but sometimes the latter is faster for exploration (if you can quickly enter
   unicode).
   
This histogram looks reasonable, but it is hard to tell if the distribution is perfect
because we did not include many points.  You can play with `Ns` and it will look better
with `Ns=5000` for example.

:::{admonition} Do It! How can you check if these are equivalent?
:class: dropdown

Formally this falls under Chapter 14.3 "Are Two Distributions Different?" of
{cite:p}`PTVF:2007`.  Another approach here is to provide "error bars" for the results
in each bin.  A good exercise is to use the fact that the probability of a sample
falling in a bin $a<\eta < b$ is
\begin{gather*}
  P_{ab} = \int_{a}^{b}p(\eta)\d{\eta} = \erf(b) - \erf(a).
\end{gather*}
Using this, you can estimate the distribution getting $N_{ab} \approx
$P_{ab}N_s$: the number of samples in this bin, and then provide an error bar.
:::

Here we simply use brute-force, generating several histograms (with the same bins) and
then using the standard deviation of these get our error bars.  Alternatively, we can
use the sophisticated {func}`matplotlib.pyplot.boxplot` and
{func}`matplotlib.pyplot.violinplot` options to show the actual distributions obtained.
These require quite a bit of customization though.

:::{margin}
From these plots it should be clear that the histogram is consistent with the underlying
distribution.  For the top plot, you might need to calibrate your eyes: these are 1σ
error-bands, so they should overlap about 68% of the time.
:::
```{code-cell}
h, bins, patches = hist

# Midpoints of the bins
positions = (bins[1:] + bins[:-1])/2

Nh = 100
# Generate Nh histograms to estimate the errors
hists = np.array(
    [np.histogram(rng.normal(scale=sigma, size=Ns)/sigma, 
                  bins=bins, density=True)[0]
     for n in range(Nh)])

fig, axs = plt.subplots(3, 1, figsize=(7, 2*3), 
                        sharex=True, gridspec_kw=dict(hspace=0.02))
for ax in axs:
    hist = ax.hist(e/sigma, **hist_kw)
    ax.plot(x, sp.stats.norm().pdf(x))

ax.set(xlim=xlim, xlabel="$e/\sigma$");

# Errorbars
axs[0].errorbar(x=positions, y=h, yerr=hists.std(axis=0), 
                fmt="+C0", alpha=0.5)

# Boxplot
axs[1].boxplot(
    hists, positions=positions, manage_ticks=False, 
    showfliers=False,  widths=np.diff(positions).mean()/4)

# Violinplot
axs[2].violinplot(
    hists, positions=positions, showextrema=False,
    widths=np.diff(positions).mean()/2);
```

This is the proper thing to do, but is generally overkill, and expensive if generating
the data is slow.  We can calculate the errors analytically as follows. Suppose that a
given bin has total probability $p$.  If we randomly choose $N$ total points, then the
probability of exactly $n$ points landing in the bin is given by the [binomial distribution][]:
\begin{gather*}
  p_n = {N \choose n} p^{n}(1-p)^{N-n}.
\end{gather*}
The mean of this is $n \approx pN$ and the variance is $N p(1-p)$.

When we use `density=True` and estimate the [PDF][], then the value of each point is
good approximation for $h \approx p \delta$ where $\delta$ is the width of the bin:
\begin{gather*}
  \d{n} \approx \sqrt{Np(1-p)}, \quad
  \d{p} \approx \sqrt{p(1-p)/N}, \quad
  \d{h} \approx \sqrt{p(1-p)/N}/\delta.
\end{gather*}

For large values of $N$, $p=n/N \ll 1$ so we can often neglect the second term.  This
gives the familiar estimate of $\d{n} \approx\sqrt{n}$ from Poisson statistics, which
must be corrected by a factor of $\d{h} = \d{n}/N\delta$ for the distribution function:
```{code-cell}
from scipy.stats import norm, binom

h, bins, patches = hist
dbin = np.diff(bins)
p = h*dbin
n = p*Ns
positions = (bins[1:] + bins[:-1])/2
dp = np.sqrt(p*(1-p)/Ns)
dh = dp / dbin
dn_poisson = np.sqrt(n)
dh_poisson = dn_poisson / Ns / dbin
fig, ax = plt.subplots(figsize=(7, 2))
hist = ax.hist(e/sigma, **hist_kw)
ax.plot(x, sp.stats.norm().pdf(x))
ax.set(xlim=xlim, xlabel="$e/\sigma$");

percentiles = 100 * norm().cdf([-1, 0, 1])
h_l, h_m, h_h = np.percentile(hists, percentiles, axis=0)
dh_hist = [h_m-h_l, h_h-h_m]

ax.errorbar(x=positions-dbin/2, y=h, yerr=hists.std(axis=0), 
           fmt="+C4", alpha=0.5, label='computed (std)')
ax.errorbar(x=positions-dbin/4, y=h, yerr=dh_hist, 
           fmt="+C0", alpha=0.5, label='computed (percentile)')
ax.errorbar(x=positions, y=h, yerr=dh, fmt="+C2", alpha=0.5, 
            label='binomial')
ax.errorbar(x=positions+dbin/4, y=h, yerr=dh_poisson, 
            fmt="+C3", alpha=0.5, label='Poisson')
ax.legend();
```

### Errors for a Weighted Histogram

The weights just change the scaling: here is some working code that should be explained,
tested, and put away.

```{code-cell}
from scipy.stats import norm, binom
rng = np.random.default_rng(seed=2)
Na = 100
Ns = 5000
w = rng.random(size=Na)*0.1
a = rng.normal(size=Na)
xlim = (-2, 2)
kw = dict(weights=w, range=xlim)
N = np.sum((xlim[0] < a) & (a <= xlim[1]))
n, bins = np.histogram(a, density=False, **kw)

# Make bins uneven
kw['bins'] = bins = bins[0] + np.linspace(0, 1, len(bins))**4 * (bins[-1] - bins[0])
n, bins = np.histogram(a, density=False, **kw)
x = 0.5 * (bins[1:] + bins[:-1])
dx = np.diff(bins)

h, b = np.histogram(a, density=True, **kw)
p = h * dx
assert np.allclose(b, bins)
assert np.allclose(n, sum(n)*p)

percentiles = 100 * norm().cdf([-1, 0, 1])
hs = []
ns = []
for _n in range(Ns):
    a_ = rng.normal(size=Na)
    h_, b = np.histogram(a_, density=True, **kw)
    n_, b = np.histogram(a_, density=False, **kw)
    hs.append(h_)
    ns.append(n_)
hs, ns = map(np.array, (hs, ns))
dhs = np.percentile(hs, percentiles, axis=0)
dns = np.percentile(ns, percentiles, axis=0)
dps = binom(N, p).ppf(percentiles[:, None]/100)/N
dh = np.array([dhs[1] - dhs[0], dhs[2] - dhs[1]])
dn = np.array([dns[1] - dns[0], dns[2] - dns[1]])
dp = np.array([dps[1] - dps[0], dps[2] - dps[1]])

plt.close('all')
fig, axs = plt.subplots(2, 2)
ax = axs[0, 0]
stairs_ = ax.stairs(h, bins)
ax.errorbar(x, h, yerr=dh, linestyle="none", c=stairs_.get_ec())
ax.errorbar(x+0.1, h, yerr=dp/dx, linestyle="none")
ax.errorbar(x+0.2, h, yerr=sum(h*dx)*dp/dx, linestyle="none")
ax = axs[0, 1]
stairs_ = ax.stairs(n, bins)
ax.errorbar(x, n, yerr=dn, linestyle="none", c=stairs_.get_ec())
_ = ax.errorbar(x+0.1, n, yerr=sum(n)*dp, linestyle="none")

# We can do this with our custom hist_err function
from math_583.histogram import histerr
plt.sca(axs[1, 0])
hist_err(a, bins=bins, density=True)
plt.sca(axs[1, 1])
_ = hist_err(a, bins=bins, density=False)
```

### Cumulative Distribution Function

While we are on the topic, note that histograms are very subjective, and depend
sensitively on how the bins are chosen.  If you actually want to compare data, it is
much better to use the **empirical distribution function** or empirical cumulative
distribution function ([eCDF][]).

```{code-cell}
def get_ecdf(x):
    """Return (x, ecdf), the eCDF for x."""
    x = np.sort(x)
    ecdf = np.arange(1, len(x)+1) / len(x)
    return (x, ecdf)

x, ecdf = get_ecdf(e/sigma)

fig, ax = plt.subplots()
ax.plot(x, ecdf)
ax.plot(x, sp.stats.norm().cdf(x))
ax.set(xlim=xlim, xlabel="$e/\sigma$");
```

### Kolmogorov-Smirnov Test

:::{margin}
To see that binning loses information, consider one single bin with the entire domain.
No matter what we do, we will get $N$ counts in this bin.  Thus, the histogram has no
errors, but tells us nothing about the underlying distribution.
:::
How can we quantitatively test if the data came from a specific distribution?  This is
the subject of Chapter 14.3 "Are Two Distributions Different?" in {cite:p}`PTVF:2007`.
The histogram error-bars discussed above are related to the discussion there about the
chi-square test, but it is better to work directly with the [eCDF][], which does not
lose any information by binning.  The [CDF][] is used in the [Kolmogorov-Smirnov test][]
of the null hypothesis that the sample comes from the underlying distribution.  It can
be computed using {func}`scipy.stats.kstest`:

```{code-cell}
res = sp.stats.kstest(e/sigma, sp.stats.norm().cdf)
res.pvalue
```

If the `p`-value is less than $p<1-c/100$, then you can rule out this hypothesis with
confidence of $c$.  I.e. if $p < 0.05$, then you can claim that the data did not come
from the distribution with 95% confidence.  Mode specifically: if we sampled the given
distribution, then we will find $p < c/100$ $c$ percent of the time.

Don't trust me?  Good!  Let's check!

```{code-cell}
from math_583.histogram import hist_err
N = 100
Ns = 5000
rng = np.random.default_rng(seed=2)
cdf = sp.stats.norm().cdf

fig, ax = plt.subplots()
for n, N in enumerate([10, 100, 1000]):
    ps = [sp.stats.kstest(rng.normal(size=N), cdf).pvalue
          for n in range(Ns)]
    hist_err(ps, **hist_kw, color=f"C{n}", label=f"{N=} points")
ax.set(
    title=r"Distribution of $p$-values for the KS test of $\mathcal{N}(0, 0)$",
    xlabel="$p$")
_ = ax.legend()

```

Note: this looks like a uniform distribution as advertised.  Is it consistent with a
uniform distribution?  We could try applying the [Kolmogorov-Smirnov test][]...

[Kolmogorov-Smirnov test]: <https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test>

### Confidence Intervals

One final comment about distributions.  Once we know the final distribution, one would
generally like to report it concisely.  This is done in terms of [confidence regions][]
or confidence intervals.  At least, one should specify a percentage: e.g. a 95%
confidence interval should contain 95% of the data.  This is not unique, however, since
there are many ways of partitioning the data into a 95% region.

With 1D data, a natural choice is to make this symmetric in the [eCDF][]: i.e. the
region between the 2.5th percentile and the 97.5th percentile.
In higher dimensions, the [eCDF][] is not well-defined -- which direction would one
"accumulate" in?  Thus, a different choice is usually made in terms of contours of the
[PDF][].  For skewed distributions, these two choices are not the same.

Finally: one often finds [confidence regions][] expressed in terms of 1σ, 2σ, 3σ, etc..  In 1D
these correspond to 68.27%, 95.45%, 99.73%, regions etc.  These numbers can be computed
using the [CDF][] for a normal distribution:
\begin{align*}
  \erf(1) - \erf(-1) &= 68.27\%, \\
  \erf(2) - \erf(-2) &= 95.45\%, \\
  \erf(3) - \erf(-3) &= 99.73\%.
\end{align*}

```{code-cell}
cdf = sp.stats.norm().cdf
for n in range(1, 6):
    p = (cdf(n)-cdf(-n))
    print(f"{n}σ = {100*p:.5f}%, 1 in {int(np.round(1/(1-p),0))}")
```
A discovery announced at a given confidence level has a 1 in $x$ chance of being
incorrect where $x$ is shown above: 5σ is one-in-a-million.

:::{margin}
The standard multi-normal distribution in $d$-dimensions has [PDF][]:
\begin{gather*}
  p(\vect{r}) = \frac{e^{-r^2/2}}{\sqrt{(2\pi)^d}}.
\end{gather*}
When averaged over the angles and considered as a function of $r$, this becomes the [chi
distribution][]:
\begin{gather*}
  p(r) = \int p(\vect{r})\d\Omega_d \\
  = \frac{r^{d-1}e^{-r^2/2}}{2^{d/2-1}\Gamma(\frac{d}{2})}.
\end{gather*}
:::
In higher dimensions, one must be cautious with this notation, and I would recommend
always specifying the confidence region in terms of a percentage.  The source of
confusion is that the ellipse of points distance 1σ from the mean in $d>1$-dimensions
contains much less than 68.27%:
```{code-cell}
for dim in [1, 2, 3, 4]:
    chi = sp.stats.chi(df=dim)
    print(f"{dim=}: 1σ contour encloses {100*chi.cdf(1):.4f}%")
```
Thus, to be clear, state "a 68% confidence region" rather than 1σ.

# Single Value Examples

For our first example, suppose that our experiment is well modeled by
\begin{gather*}
  y_n = (\sqrt{a} + \eta_n)^2 = a + e_n
\end{gather*}
where $\eta_n \sim \mathcal{N}(0, \sigma)$ are normally distributed with [PDF][]
\begin{gather*}
    p(\eta) = \frac{e^{-\eta^2/2\sigma^2}}{\sqrt{2\pi}\sigma}.
\end{gather*}
This form produces a non-gaussian distribution for the errors $e_n$ in our measured
quantity $y_n \equiv y$.

:::{margin}
One needs to be careful about multiple solutions.  I.e., when $\eta < \sqrt{a}$.  We
deal with this below when computing $p_e(e)$ by explicitly summing over all solutions
$\eta_{\pm}$.
:::
Let's look at how these errors are distributed.  One can do this analytically by noting that
\begin{gather*}
  e = (\sqrt{a} + \eta)^2 - a, \qquad
  \eta = \sqrt{a + e} - \sqrt{a}.
\end{gather*}
Since probability is conserved:
\begin{gather*}
  \int p_e(e;a)\d{e} = \int p(\eta)\d{\eta},
\end{gather*}
as long as the domains match.  Thus, we can derive $p_e(e;a)$ by differentiating:
\begin{gather*}
  \underbrace{\int_0^\eta p(\eta)\d{\eta}}_{\erf(\eta)} = 
  \int_{0}^{e} p_e(e;a)\d{e}, \qquad
  p(\eta) = \diff{e}{\eta} p_e(e;a),\\
  p_e(e;a) = \sum_{\eta_{\pm}}\diff{e}{\eta}p(\eta)
           = \sum_{\pm}\frac{1}{2\sqrt{a+e}}p(\pm\sqrt{a+e}-\sqrt{a}).
\end{gather*}

This analytic work is a pain, however, and prone to errors, so instead, let's just
generate samples.  We can then use this to test our expression:
```{code-cell}
rng = np.random.default_rng(seed=2)
norm = sp.stats.norm()

Ns = 4000
eta = rng.normal(size=Ns)

hist_kw = dict(bins=100, density=True, histtype='step')
fig, ax  = plt.subplots(figsize=(7, 4))
for n, a in enumerate([0, 1, 2, 3]):
    e = (np.sqrt(a) + eta)**2 - a
    xlim = np.percentile(e, [0.1, 80])
    plt.sca(ax)
    h, b, _ = hist_err(e, **hist_kw, label=f"{a=}", color=f"C{n}")
    db = np.diff(b)
    p = h*db
    dh = np.sqrt(p*(1-p)/Ns)/db
    ax.errorbar(b[:-1]+db/2, h, yerr=dh, fmt=f"+C{n}", alpha=0.5)
    e_ = np.linspace(*xlim, 500)
    eta1_ = np.sqrt(a+e_) - np.sqrt(a)
    eta2_ = -np.sqrt(a+e_) - np.sqrt(a)
    pe_ = sum(norm.pdf(eta_)/2/np.sqrt(a+e_) for eta_ in (eta1_, eta2_))
    ax.plot(e_, pe_, f"--C{n}")
    ax.legend()
_ = ax.set(xlabel="e", ylim=(0, 1.25), xlim=xlim)
```

While this is useful, and we see that these errors are highly non-gaussian, it is
unnecessary.  We can simply try calibrating the mean $A(\vect{y}) = \bar{y}$ as an
estimator over the range $a \in [0, 5]$ when we have $N$ data points.  We will include
the 68% confidence bands.

```{code-cell}
rng = np.random.default_rng(seed=2)

def get_A(y):
    """Return the estimator."""
    return np.mean(y)
    
N = 100
Ns = 1000
as_ = np.linspace(0, 5)
cl = 68.0
p = (100- cl)/2
percentiles = [p, 50, 100-p]
calibration = []
for a in as_:
    As = [get_A((np.sqrt(a) + rng.normal(size=N))**2)
          for n in range(Ns)]
    As_ = np.percentile(As, percentiles)
    A_ = np.mean(As)
    calibration.append([A_] + As_.tolist())

calibration = np.array(calibration)
mean, l, median, h = calibration.T
fig, ax  = plt.subplots(figsize=(7, 4))
ax.plot(as_, mean, 'C0', label="mean")
ax.plot(as_, l, ':C1')
ax.plot(as_, median, 'C1', label="median")
ax.plot(as_, h, ':C1')
ax.grid('on')
ax.set(title=f"{N=} calibration with {cl}% confidence region", xlabel="a", ylabel="A")
_ = ax.legend()
```

In this case where $p(\eta)$ is a normal distribution, we see that we have a very nice
linear relationship with a systematic offset of 1.  I.e., if the mean is $\bar{A}$, then
the median estimate for $a \approx \bar{A} - 1$.

:::{margin}
Note: Spline interpolation requires that the abscissa be monotonically increasing.  If
the calibration curve has negative slope, then you will need to reverse the arguments.
This methods will naturally fail if the function is not monotonic, but in that case,
special treatment is needed since the confidence region will be disjoint.
:::
To use this practically, one might fit a spline or a polynomial.  Here we use the median
of the calibration samples for calibration:
```{code-cell}
from scipy.interpolate import InterpolatedUnivariateSpline

As_ = median

# We have A, l, h as functions of a
# We want a, l, h as functions of A
# To do this, we form the latter by flipping the arguments
# interpolate l(a(A)) and h(a(A))
a_A = InterpolatedUnivariateSpline(As_, as_)
l_A = InterpolatedUnivariateSpline(l, as_)
h_A = InterpolatedUnivariateSpline(h, as_)

fig, ax = plt.subplots()

ax.plot(as_, l, ':C0')
ax.plot(as_, median, 'C0', label="data")
ax.plot(as_, h, ':C0')

ax.plot(l_A(As_), As_, '-.C1')
ax.plot(a_A(As_), As_, '--C1', label="spline")
ax.plot(h_A(As_), As_, '-.C1')

ax.set(title=f"{N=} calibration", xlabel="a", ylabel="A")
_ = ax.legend()
```
The orange curves are computed as functions of the estimator $A$.  Note: we must not
trust the splines outside the region of interpolation.

We have packaged this into a custom routine {func}`math_583.fitting.calibrate`:

```{code-cell}
from math_583.fitting import calibrate

rng = np.random.default_rng(seed=2)
N = 100

def get_estimation(a, N=N, rng=rng):
    y = (np.sqrt(a) + rng.normal(size=N))**2
    return y.mean()

a = np.linspace(0, 5)
calibration = calibrate(get_estimation, a=a)

fig, ax = plt.subplots()
labels = ["mean", "-1σ", "+1σ"]
for c, label in zip(calibration, labels):
    ax.plot(a, c, label=label)
ax.set(xlabel="$a$", ylabel="$A$")
_ = ax.legend()
```


### Sample Variance

:::{margin}
Mathematically you can trace this to the fact that we use $\bar{y}$, the sample mean,
rather than the true mean in the calculation.
:::
As an example we test the familiar formula for the variance of a sample:
\begin{gather*}
    \sigma^2(\vect{y}) = \frac{\sum_{n}(y_n - \bar{y})^2}{N-1} 
    = \frac{\sum y_n^2 - N\bar{y}^2}{N-1}, \qquad
    \bar{y} = \frac{\sum_{n} y_n}{N}.
\end{gather*}
Here the denominator for the variance is $N-1$ which differs from the population
variance where it is $N$.  Let's see if this is correct by calibrating the two
estimators:

```{code-cell}
from math_583.fitting import calibrate

rng = np.random.default_rng(seed=2)

N = 3       # Keep it small so we can see
mean = 1.2

def population(sigma):
    """Estimate the variance using the wrong formula."""
    y = rng.normal(loc=mean, scale=sigma, size=N)
    return sum((y - y.mean())**2)/N

def sample(sigma):
    """Estimate the variance using the wrong formula."""
    y = rng.normal(loc=mean, scale=sigma, size=N)
    return sum((y - y.mean())**2)/(N-1)

sigmas = np.linspace(0.5, 2)

# Note: it is important to use the mean here:
kw = dict(use_mean=True, sigma_levels=None)
cps = calibrate(population, sigmas, **kw)
css = calibrate(sample, sigmas, **kw)
fig, ax = plt.subplots()
for n, (cp, cs) in enumerate(zip(cps, css)):
    ax.plot(sigmas**2, cp, '--C0', label="Population estimator" if n == 0 else "")
    ax.plot(sigmas**2, cs, '--C1', label="Sample estimator" if n == 0 else "")
ax.plot(sigmas**2, sigmas**2, ':C2', label="exact")
ax.set(xlabel="Population Variance", ylabel="estimation")
_ = ax.legend()
```

These notes closely follow Chapter 15.6 "Confidence Limits on Estimated Model
Parameters" of {cite:p}`PTVF:2007` which I highly recommend you read in parallel.

[chi-squared distribution]: <https://en.wikipedia.org/wiki/Chi-squared_distribution>
[chi distribution]: <https://en.wikipedia.org/wiki/Chi_distribution>
[sum or normally distributed random variables]: <https://en.wikipedia.org/wiki/Sum_of_normally_distributed_random_variables>
[covariance matrix]: <https://en.wikipedia.org/wiki/Covariance_matrix>
[uncertainties]: <https://pythonhosted.org/uncertainties/>
[`collections.namedtuple`]: <https://docs.python.org/3/library/collections.html#collections.namedtuple>
[confidence regions]: <https://en.wikipedia.org/wiki/Confidence_region>
[nuisance parameter]: <https://en.wikipedia.org/wiki/Nuisance_parameter>
[least squares]: <https://en.wikipedia.org/wiki/Least_squares>
[algebra of random varables]: <https://en.wikipedia.org/wiki/Algebra_of_random_variables>
[principal componant analysis]: <https://en.wikipedia.org/wiki/Principal_component_analysis>
[reduced chi-square statistic]: <https://en.wikipedia.org/wiki/Reduced_chi-squared_statistic>
[multivariate normal distribution]: <https://en.wikipedia.org/wiki/Multivariate_normal_distribution>
[likelihood function]: <https://en.wikipedia.org/wiki/Likelihood_function>
[Bayesian inference]: <https://en.wikipedia.org/wiki/Bayesian_inference>
[Bayes' theorem]: <https://en.wikipedia.org/wiki/Bayes%27_theorem>
[model evidence]: <https://en.wikipedia.org/wiki/Marginal_likelihood>
[maximum likelihood estimation]: <https://en.wikipedia.org/wiki/Maximum_likelihood_estimation>
[Poisson distribution]: <https://en.wikipedia.org/wiki/Poisson_distribution>
[mean]: <https://en.wikipedia.org/wiki/Expected_value>
[median]: <https://en.wikipedia.org/wiki/Median>
[mode]: <https://en.wikipedia.org/wiki/Mode_(statistics)>
[standard deviation]: <https://en.wikipedia.org/wiki/Standard_deviation>
[probability distribution]: <https://en.wikipedia.org/wiki/Probability_distribution>
[skewness]: <https://en.wikipedia.org/wiki/Skewness>
[kurtosis]: <https://en.wikipedia.org/wiki/Kurtosis>
[marginal distributions]: <https://en.wikipedia.org/wiki/Marginal_distribution>
[bootstrapping]: <https://en.wikipedia.org/wiki/Bootstrapping_%28statistics%29>
[calibration]: <https://en.wikipedia.org/wiki/Calibration_(statistics)>
[estimator]: <https://en.wikipedia.org/wiki/Estimator>
[MCMC]: <https://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo>
[Hessian matrix]: <https://en.wikipedia.org/wiki/Hessian_matrix>
[PDF]: <https://en.wikipedia.org/wiki/Probability_density_function>
[binomial distribution]: <https://en.wikipedia.org/wiki/Binomial_distribution>
[eCDF]: <https://en.wikipedia.org/wiki/Empirical_distribution_function>
[CDF]: <https://en.wikipedia.org/wiki/Cumulative_distribution_function>
