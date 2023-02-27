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

(sec:GraphLaplacian)=
# Graph Laplacian

## Graphs

:::{margin}
I am largely using the notations defined by [Naoki
Saito](https://www.math.ucdavis.edu/~saito/courses/HarmGraph/lectures.html), especially
[Lecture 03](https://www.math.ucdavis.edu/~saito/courses/HarmGraph/lecture03.pdf).
:::
We start with some preliminary notations about graphs.  A graph $G = (V, E)$ is a collection of
vertices $V$ and edges $E \subset V\times V$.  We can represent a graph numerically with
its **adjacency matrix** $\mat{A}$ by first labeling the vertices by an index $i$: $v_i
\in V$ such that we can refer to the vertex by $i$.  Thus, we will refer to the edge $e
= (v_i, v_j) \equiv (i, j)$.  The (unweighted) adjacency matrix $\mat{A}$ is
\begin{gather*}
  [\mat{A}]_{ij} = a_{ij} = \begin{cases}
    1 & \text{if } (v_i, v_j) \in E,\\
    0 & \text{otherwise.}
  \end{cases}
\end{gather*}
:::{margin}
Note: some references, like {cite:p}`Bauer:2012`. use the transpose $w_{ji}$ for the
weight corresponding to the edge $e=(i,j)$.  They claim this is more natural for working with
dynamical systems.  To compare with these works, use $\mat{A}^T = \mat{W}^T$ where they
use $\mat{A}$.
:::
Sometimes we want to associate a **weight** $w_{ij}$ with each edge $e=(i, j)$.  This might codify
a channel capacity (used in calculating the {ref}`sec:FlatNorm`), or it might be
associated with a distance $d_{ij} = d(v_i, v_j)$ between the vertices.  In this case,
we have a **weighted adjacency matrix** $\mat{A}=\mat{W}$.

:::{note}
Unless the graph is almost fully connected, the adjacency matrix will be sparse,
thus we usually use a sparse matrix representation, such as provided by
{mod}`scipy.sparse`.  This provides one more option: one can ask if the sparse matrix
$\mat{A}$ contains an entry $a_{ij}$, which is different that asking if $a_{ij} \neq
0$.  I.e., we can retain adjacency information even if an edge has zero weight.  Masked
arrays can also be used.  These are used by the algorithms in
{mod}`scipy.sparse.csgraph` which will generally be our first consideration when selecting
algorithms.
:::

Note that the adjacency matrix need not be symmetric unless the graph is symmetric.  The
adjacency matrix has some nice properties.

1. Powers of the unweighted adjacency matrix $\mat{A}^n$ describe how many paths of
   length $n$ connect different vertices.

Related to the adjacency matrix $\mat{A}$ are the degrees $d_i$ of each vertex:
\begin{align*}
  d^{\text{out}}_j &= \sum_{i}A_{ij}, \\
  d^{\text{in}}_i &= \sum_{j}A_{ij}.
\end{align*}
For undirected graphs we have simply $d_{i} = d^{\text{in}}_i = d^{\text{out}}_i$.  From
these we can construct the **degree matrix** $\mat{D}$:
\begin{gather*}
  \mat{D} = \diag(\vect{d}^{\text{out}}).
\end{gather*}

According to {cite:p}`Bauer:2012`, the normalized graph Laplacian is
\begin{gather*}
  \mat{\bar{\Delta}} = \mat{1} - \mat{D}^{-1}\mat{A}.
\end{gather*}
with appropriate rows set to zero if $d_{i}^{\text{out}} = 0$.





* The **degree matrix** $\mat{D}$, which is the diagonal matrix who's entries are the
  sum of the rows of $\mat{A}$: 

* The [**transition matrix**](https://en.wikipedia.org/wiki/Stochastic_matrix)
  $\mat{P}$, which is the diagonal matrix who's entries are:
  \begin{gather*}
    \mat{P} = \mat{D}^{-1}\mat{A}, \qquad
    P_{ij} = \frac{A_{ij}}{d_i}.
  \end{gather*}
  $[\mat{P}^n]_{ij}$ represents the probability of a random walk from $v_i$ to $v_j$ in
  $n$ steps starting with probabilities $P_{ij}$ for a single step.  These probabilities
  are relative to the edge weights, suitably normalized so that $\sum_j P_{ij} = 1$.


### Symmetrization

Include both $(v_i, v_j)$ and $(v_i, v_j)$:
* **$k$-nearest neighbour graph**: if *either* $v_i$ is near to $v_j$ *or* $v_j$ is near $v_i$.
* **mutual $k$-nearest neighbour graph**: if *both* $v_i$ is near to $v_j$ *and*
  $v_j$ is near to $v_i$.

## Laplacian

We start by considering the Laplacian in $\mathbb{R}^n$:
\begin{gather*}
  \nabla^2 f(x) = \sum_{i} \pdiff[2]{f(\vect{x})}{x_i}.
\end{gather*}

The Laplacian has some nice features:
1. It is a linear operator.  *Note: this depends on the boundary conditions, but this
   remains true for periodic; Dirichlet, Neumann, and Robin with zero values; and
   various reflections.*
2. The eigenfunctions are plane-waves:
   \begin{gather*}
     \nabla^2 e^{\I \vect{k}\cdot{x}} = -k^2 e^{\I \vect{k}\cdot{x}}.
   \end{gather*}
   The eigenfunctions can thus be used as a Fourier basis.
   




Now consider tabulating $f_n = f(x_n)$ at a set of points $x_n$.  How can 




```{code-cell} ipython3
%matplotlib inline
import numpy as np, matplotlib.pyplot as plt
import scipy.stats
sp = scipy

def in_region(x, y, x0=-5.0, x1=5.0, r0=4, r1=3, d=1):
    """Define a region of interest."""
    z = x + 1j*y
    return ((abs(y)<d/2) & (x0<x) & (x<x1) | (abs(z-x0)<r0) | (abs(z-x1)<r1))
```

```{code-cell} ipython3
xlim = (-10, 10)
ylim = (-5, 5)
x_ = np.linspace(*xlim, 256)
y_ = np.linspace(*ylim, 256//2)
X, Y = np.meshgrid(x_, y_, indexing='ij', sparse=True)
plt.imshow(in_region(X, Y).T)
```

```{code-cell} ipython3
rng = np.random.default_rng(seed=2)
Ng = 400  # Size of graph
xy = []
while len(xy) < Ng:
    _x = xlim[0] + np.diff(xlim)[0] * rng.random()
    _y = ylim[0] + np.diff(ylim)[0] * rng.random()
    if in_region(_x, _y):
        xy.append((_x, _y))
xy = np.asarray(xy)
```

```{code-cell} ipython3
fig, ax = plt.subplots()
ax.pcolormesh(x_, y_, in_region(X, Y).T, shading='auto')
ax.plot(*zip(*xy), '+')
ax.set(aspect=1)
```

We will use [`scipy.spatial.KDTree`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.html) to find the nearest neighbours and then build the adjacency graph.

```{code-cell} ipython3
import scipy.spatial, scipy.sparse.csgraph
sp = scipy

neighbours = 6
tree = sp.spatial.KDTree(xy)
N = len(xy)
G = sp.sparse.lil_matrix((N, N))
graph = dict
for n0 in range(N):
    ds, ns = tree.query(xy[n0], k=neighbours)
    assert ds[0] == 0
    ds, ns = ds[1:], ns[1:]  # Skip the node itself
    #G[(n0, n0)] = 0len(ns)
    for n1, d in zip(ns, ds):  
        G[n0, n1] = 1
```

```{code-cell} ipython3
fig, ax = plt.subplots()
ax.pcolormesh(x_, y_, in_region(X, Y).T, shading='auto')
ax.set(aspect=1)
for i0, i1 in zip(*G.nonzero()):
    if i0 == i1:
        continue
    ax.plot(*xy[[i0,i1]].T, '-+')
```

```{code-cell} ipython3
# Simple version of the Laplacian as the symmetrized adjacency matrix
L2 = ((G.T + G)/2).toarray() > 0
L2 = L2 - np.diag(L2.sum(axis=0))
assert np.allclose(L2, L2.T)
d, V = np.linalg.eigh(L2)
inds = np.argsort(abs(d))
d, V = d[inds], V[:, inds]
plt.plot(V[:, 0])
```

```{code-cell} ipython3
x, y = xy.T
F = np.exp(-(X**2+Y**2)/5**2/2)
f = np.exp(-(x**2+y**2)/5**2/2)


fig, ax = plt.subplots()
ax.pcolormesh(x_, y_, (F*in_region(X, Y)).T, shading='auto')
ax.plot(*zip(*xy), '+')
ax.set(aspect=1)
```

```{code-cell} ipython3
N = 10
print(len(f))
abs(V[:, :N] @ (V.T @ f)[:N] - f).max()
```

```{code-cell} ipython3
x, y = xy.T
F = np.exp(-(X**2+Y**2)/5**2/2)
f = np.exp(-(x**2+y**2)/5**2/2)


fig, ax = plt.subplots()
ax.pcolormesh(x_, y_, (F*in_region(X, Y)).T, shading='auto')
ax.plot(*zip(*xy), '+')
ax.set(aspect=1)
```

```{code-cell} ipython3
print(len(f))
abs(V[:, inds][:, :N] @ (V[:, inds].T @ f)[:N] - f).max()
```

```{code-cell} ipython3

```
