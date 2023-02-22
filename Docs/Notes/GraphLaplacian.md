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

```{code-cell} ipython3
np.logical_or?
```

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
x = np.linspace(*xlim, 256)
y = np.linspace(*ylim, 256//2)
X, Y = np.meshgrid(x, y, indexing='ij', sparse=True)
plt.imshow(in_region(X, Y).T)
```

```{code-cell} ipython3
rng = np.random.default_rng(seed=2)
```

```{code-cell} ipython3
np.diff(xlim)[0] * rng.random()
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
ax.pcolormesh(x, y, in_region(X, Y).T, shading='auto')
ax.plot(*zip(*xy), '+')
ax.set(aspect=1)
```

We will use [`scipy.spatial.KDTree`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.html) to find the nearest neighbours and then build the adjacency graph.

```{code-cell} ipython3
import scipy.spatial, scipy.sparse.csgraph
sp = scipy

neighbours = 4
tree = sp.spatial.KDTree(xy)
N = len(xy)
G = sp.sparse.lil_matrix((N, N))
graph = dict
for n0 in range(N):
    ds, ns = tree.query(xy[n0], k=neighbours)
    assert ds[0] == 0
    ds, ns = ds[1:], ns[1:]  # Skip the node itself
    G[(n0, n0)] = len(ns)
    for n1, d in zip(ns, ds):  
        G[n0, n1] = 1
```

```{code-cell} ipython3
xy[i0], xy[i1]
i0, i1
```

```{code-cell} ipython3
fig, ax = plt.subplots()
ax.pcolormesh(x, y, in_region(X, Y).T, shading='auto')
ax.set(aspect=1)
for i0, i1 in zip(*G.nonzero()):
    if i0 == i1:
        continue
    ax.plot(xy[i0], xy[i1], '+')
```

```{code-cell} ipython3
L2 = sp.sparse.csgraph.laplacian(G)
assert (L2 - L2).nnz == 0
d, V = np.linalg.eigh(L2.toarray())
inds = np.argsort(abs(d))
plt.plot(V[:,inds][:, 2]),  d[inds][2]
```

```{code-cell} ipython3

```
