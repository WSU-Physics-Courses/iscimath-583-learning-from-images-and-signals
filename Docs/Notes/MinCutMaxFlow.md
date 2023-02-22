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

```{code-cell} ipython3
import numpy as np
from importlib import reload
from math_583 import denoise; reload(denoise)
im = denoise.Image()
u_noise = im.get_data(sigma=0.4)
l1tv = denoise.L1TVMaxFlow(u_noise, laminv2=2)
im.show(l1tv.denoise1(threshold=0.2), u_noise, im.get_data(sigma=0))
```

```{code-cell} ipython3
%time im.show(l1tv.denoise(N=10), u_noise, im.get_data(sigma=0))
```

```{code-cell} ipython3
%%time
d = denoise.Denoise(im, p=1, q=1, lam=2/2)
u = d.minimize(callback=None)
```

```{code-cell} ipython3
%%time
ths = np.linspace(u_noise.min(), u_noise.max(), 10)
us = list(map(l1tv.denoise1, ths))
im.show(*us)
```

```{code-cell} ipython3
im.show(u)
```

# L1TV

```{code-cell} ipython3
import numpy as np
import maxflow
from math_583 import denoise
im = denoise.Image()
u_noise = im.get_data()
threshold = 0.5  # Threshold
lam = 0.2        # Curvature
g = maxflow.Graph[float]()
nodeids = g.add_grid_nodes(u_noise.shape)
a, b, c = 0.1221, 0.0476, 0.0454
structure = np.array([
    [0, c, 0, c, 0],
    [c, b, a, b, c],
    [0, a, 0, a, 0],
    [c, b, a, b, c],
    [0, c, 0, c, 0]])
g.add_grid_edges(nodeids, structure=structure, symmetric=True)

# I would like to store this graph, and then make copies for processing
# with different thresholds and smoothing parameters:

lam = 0.2
threshold = 0.5
#g1 = g.copy()   # <<<<<<<<<<<< How to do this?
g1 = g
sources = u_noise >= threshold
g1.add_grid_tedges(nodeids, lam * sources, lam * (1 - sources))
g1.maxflow()
u_clean = g.get_grid_segments(nodeids)
```

(sec:MinCutMaxFlow)=
Min-Cut/Max-Flow
================

## Network

The [min-cut max-flow theorem] relates the maximal flow through a network to the minimum
cut as follows.  Consider a directed graph $N = (V, E)$ of vertices $V$ and edges
$E\subset V\times V$ where, associated with each edge is a **capacity** $c: E\mapsto
\mathbb{R}^+$.  This graph can be represented by a weighted [adjacency matrix][]
$\mat{C}$ whose values $C_{ij}$ represent the capacity of the channels connecting vertex
$i$ with vertex $j$.  Note: if the model is of channels as pipes such that they have the
same capacity in each direction, then the matrix will be symmetric $\mat{C} = \mat{C}^T$
and the graph is **undirected**.

## Flows 

The [min-cut max-flow theorem][] considers the allowed flow $\abs{f}$ from a **source**
vertex $s$ to a **target** (sink) vertex $t$.  Think of this as pouring water into the
source $s$ at rate $\abs{f}$ and draining it from the target $t$ at the same rate.
Through each channel, the flow of water must not exceed the capacity, and the water must
be conserved.  Formally, a **flow** through the network is mapping $f: E\mapsto \mathbb{R}^+$ such that:

1. **Capacity**: $f_{ij} \leq c_{ij}$: i.e. the flow through a channel never exceeds the channel
   capacity,
2. **Conservation**: The net flow through is conserved theough each vertex, except the
   source $s$, which has an additional input flow $\abs{f}$, and target $t$, which has an
   additional output flow $\abs{f}$.
   
## Cuts

An **$s$-$t$ cut** $C = (S, T)$ is a partition of the vertices $V$ into two disjoint sets $S$ and
$T$ such that $s\in S$ and $t\in T$.  The **capacity** $c(X, T)$ of the $s$-$t$ cut is
the sum of the capacities of the edges in the cut-set $X_C$:

\begin{gather*}
  X_C = \{(i, j) \in E \mid i \in S,  j \in T \} = (S\times T) \cap E,\\
  c(S,T) = \sum_{(i, j) \in X_c} c_{ij}.
\end{gather*}

The [min-cut max-flow theorem][] relates the [maximum flow problem][] to the [minimum
$s$-$t$ cut problem][], stating that the maximum flow $\abs{f}$ is equal to the minimum
cut $c(X, T)$:

1. [maximum flow problem][]: Maximize the flow $\abs{f}$ between $s$ and $t$.
2. [minimum $s$-$t$ cut problem][]: Minimize the cut $c(S,T)$.

For our purposes, we will represent the network through the weighted [adjacency matrix]
$\mat{C}$, typically as a sparse matrix.

:::{admonition} Implementation Details

:class: dropdown

The [maximum flow problem][] can be solved quite quickly using
{func}`scipy.sparse.csgraph.maximum_flow`.  This requires that we convert $\mat{C}$ to a
{class}`scipy.sparse.csr_matrix`, which can be easily done from any sparse matrix with
the `tocsr()` method.  A further requirement is that the matrix contain integer values.
Thus, one may need to scale and round the entries.

SciPy does not currently provide a routine for producing the min cut $c(S,T)$
:::

```{code-cell} ipython3
from importlib import reload
from math_583 import flow; reload(flow)
import scipy.sparse
sp = scipy

C = np.array([[0, 16, 13,  0,  0,  0],
              [0,  0, 10, 12,  0,  0],
              [0,  4,  0,  0, 14,  0],
              [0,  0,  9,  0,  0, 20],
              [0,  0,  0,  7,  0,  4],
              [0,  0,  0,  0,  0,  0]])
f = flow.Flow(C, s=0, t=5)
cut = f.min_cut_igraph()
N = C.shape[0]
#f.pos = np.arange(N) - N/2, np.zeros(N)
pos = f.plot()
f1, fv = f.max_flow()
#f1.pos = np.arange(N) - N/2, np.zeros(N)
f1.plot(pos=pos, cut=cut)
```

## $L^1$-TV

```{code-cell} ipython3
from importlib import reload
from math_583 import flow, denoise; reload(flow); reload(denoise)
import scipy.sparse
sp = scipy

im = denoise.Image()
u_noise = im.get_data()
l1tv = denoise.L1TV(u_noise)
```

```{code-cell} ipython3
import igraph
self = l1tv
lam = 2/100
N = np.prod(self.u_noise.shape)
s, t = N, N+1

C = (self._weights + self._connections * lam).tocoo()
f = flow.Flow(C, s=s, t=t)
%time g = igraph.Graph.Weighted_Adjacency(C, mode='directed', attr="capacity")
%time res = g.st_mincut(s, t, capacity="capacity")
#%time cut = f.min_cut()
```

```{code-cell} ipython3
%load_ext line_profiler
display(im.show(u_noise))
l1tv.laminv2 = 2
%lprun -f flow.Flow.min_cut_igraph u = l1tv.denoise(laminv2=100)
display(im.show(u))
```

```{code-cell} ipython3
%load_ext line_profiler
from importlib import reload
from math_583 import flow, denoise; reload(flow); reload(denoise)
import scipy.sparse
sp = scipy

im = denoise.Image()
u_noise = im.get_data()
laminv2 = 4
%lprun -f denoise.compute_l1tv u = denoise.compute_l1tv(u_noise, laminv2=laminv2)
```

```{code-cell} ipython3
%load_ext line_profiler
from importlib import reload
from math_583 import flow, denoise; reload(flow); reload(denoise)
import scipy.sparse
sp = scipy

im = denoise.Image()
u_noise = im.get_data()
f = denoise.L1TVMaxFlow(u_noise)
u = f.denoise(laminv2=30)
u1 = denoise.compute_l1tv(u_noise, laminv2=2)
im.show(u.astype(int) - u1)
```

```{code-cell} ipython3
%load_ext line_profiler
from importlib import reload
from math_583 import flow, denoise; reload(flow); reload(denoise)
import scipy.sparse
sp = scipy

im = denoise.Image()
u_noise = im.get_data()
f = denoise.L1TVMaxFlow(u_noise)
laminv2 = 4
%lprun -f denoise.L1TVMaxFlow.denoise u = f.denoise(laminv2=laminv2)
im.show(u)
```

```{code-cell} ipython3
import copy
```

```{code-cell} ipython3
import maxflow
threshold = 0.5
laminv2 = 4
lam = 2/laminv2
g = maxflow.Graph[float]()
nodeids = g.add_grid_nodes(u_noise.shape)
a, b, c = 0.1221, 0.0476, 0.0454
structure = np.array(
    [[0, c, 0, c, 0],
     [c, b, a, b, c],
     [0, a, 0, a, 0],
     [c, b, a, b, c],
     [0, c, 0, c, 0]])
sources = (u_noise>=threshold)
g.add_grid_edges(nodeids, weights=1, structure=structure, symmetric=True)
te = g.add_grid_tedges(nodeids, lam*sources, lam*(1-sources))
g.maxflow()
sgm = g.get_grid_segments(nodeids)
im.show(sgm)
```

```{code-cell} ipython3
from importlib import reload
from math_583 import flow, denoise; reload(flow); reload(denoise)
import scipy.sparse
sp = scipy

im = denoise.Image()

class F(denoise.Base):
    image = None
    def __init__(self, image, **kw):
        super().__init__(image=image, **kw)
    
    def init(self):
        C = sp.sparse.csr_array()
```

```{code-cell} ipython3
np.square([1, 3]).sum()
```

```{code-cell} ipython3
(lambda *x: sum(x))(1, 2, 3)
```

```{code-cell} ipython3
%%time 
from itertools import product
Nx, Ny = im.shape

def _wkey(*x):
    """Key is the square of the distance"""
    return np.square(x).sum()

# 16 point stencil
w = {
    _wkey(0, 1): 0.1221,
    _wkey(1, 1): 0.0476,
    _wkey(1, 2): 0.0454,
}
# No support yet for sparse multi-dimensional arrays
C = dict()
mode = 'periodic'
mode = 'constant'
mode = 'reflect'  # Needs testing
for nx, ny in product(range(Nx), range(Ny)):
    for dx, dy in product(*[[-2, -1, 0, 2, 1]]*2):
        tx, ty = nx+dx, ny+dy
        if mode == 'periodic':
            tx, ty = tx % Nx, ty % Ny
        elif mode == 'constant':
            if tx < 0 or ty < 0 or tx >= Nx or ty >= Ny:
                continue
        elif mode == 'reflect':
            tx, ty = abs(tx), abs(ty)
            if tx >= Nx:
                tx = 2*Nx - tx - 1
            if ty >= Ny:
                ty = 2*Ny - ty - 1
        wkey = _wkey(dx, dy)
        key = (nx, ny, tx, ty)
        keyT = (tx, ty, nx, ny)
        if wkey in w and keyT not in C:
            C[key] = C[keyT] = w[wkey]
```

```{code-cell} ipython3
key = [1, 2]
key[::-1]
```

```{code-cell} ipython3
# Make a black and white image
u = im.get_data()
u0 = np.percentile(u, 50)
u = np.where(u < u0, 0, 1)
im.show(u)

def _key(nx, ny):
    return ny + nx*Ny

def get_flow(u, laminv2=20):
    lam = 2/laminv2
    rows = [_key(*_k[:2]) for _k in C]
    cols = [_key(*_k[2:]) for _k in C]
    vals = list(C.values())
    s = Nx*Ny
    t = s + 1
    for nx, ny in product(range(Nx), range(Ny)):
        key = _key(nx, ny)
        vals.append(lam)
        if u[nx, ny] == 0:
            rows.append(key)
            cols.append(t)
        elif u[nx, ny] == 1:
            key = _key(nx, ny)
            rows.append(s)
            cols.append(key)
        else:
            raise ValueError(f"u must be a charateristic function: got {u[nx, ny]=}")
            
    return flow.Flow(sp.sparse.csr_matrix((vals, (rows, cols)), shape=(Nx*Ny+2,)*2), s=s, t=t)

f = get_flow(u, laminv2=5)
cut = f.min_cut()
```

```{code-cell} ipython3
S, T = cut.S, cut.T
S.remove(f.s)
T.remove(f.t)
u = np.zeros(Nx*Ny, dtype=int)
u[T] = 1
u = u.reshape(Nx, Ny)
im.show(u)
```

```{code-cell} ipython3

```

```{code-cell} ipython3
S = cut.S
S.remove(s)
```

```{code-cell} ipython3
%time rows, cols, vals = map(list, np.transpose([(_key(*_k[:2]), _key(*_k[2:]), C[_k]) for _k in C]))
%time rows = [_key(*_k[:2] for _k in C]
```

```{code-cell} ipython3
%debug
```

```{code-cell} ipython3
C = np.array([[0, 16, 13,  0,  0,  0],
              [0,  0, 10, 12,  0,  0],
              [0,  4,  0,  0, 14,  0],
              [0,  0,  9,  0,  0, 20],
              [0,  0,  0,  7,  0,  4],
              [0,  0,  0,  0,  0,  0]])
f = flow.Flow(C, s=0, t=5)
f.min_cut_igraph()
```

```{code-cell} ipython3
f._res.cut
```

```{code-cell} ipython3
np.random.seed(3)

C = np.random.randint(256, size=(1000, 1000))
#C = (sp.sparse.random(10000, 10000, density=0.1)*256).astype(int)
f = flow.Flow(C, s=0, t=5)
fv = f.max_flow_scipy().flow_value
print(fv)

f_ = flow.Flow([[0, 0.16, 0.13, 0,    0,    0],
               [0, 0,    0.10, 0.12, 0,    0],
               [0, 0.04, 0,    0,    0.14, 0],
               [0, 0,    0.09, 0,    0,    0.20],
               [0, 0,    0,    0.07, 0,    0.04],
               [0, 0,    0,    0,    0,    0]], s=0, t=5)
n = 1e-25
fn = flow.Flow(C / n, s=0, t=5)
#factor = np.iinfo(np.int64).max ** (1/2.1) / abs(fn.C.max())
#fv_ = f.max_flow_scipy(factor=factor).flow_value * n
fv_ = fn.max_flow_scipy().flow_value * n
print(fv_)
print(abs(fv_/fv - 1))
```

```{code-cell} ipython3
C = np.array([[0, 16, 13,  0,  0,  0],
              [0,  0, 10, 12,  0,  0],
              [0,  4,  0,  0, 14,  0],
              [0,  0,  9,  0,  0, 20],
              [0,  0,  0,  7,  0,  4],
              [0,  0,  0,  0,  0,  0]])
f = flow.Flow(C, s=0, t=5)
Nx = f.C.shape[0]-2
g = ig.Graph.Weighted_Adjacency(f.C, mode="directed", attr="capacity")
ig.plot(g, layout=g.layout_circle())
```

```{code-cell} ipython3
Nx = f.C.shape[0]-2
g = ig.Graph.Weighted_Adjacency(f.C, mode="directed", attr="capacity")
visual_style = dict(
    #edge_width=0.3,
    #vertex_size=1.5,
    palette="heat",
    bbox=(600,300),
    #layout="fruchterman_reingold"
)
g.vs['x'] = np.arange(Nx)
g.vs['y'] = 0
g.vs['label'] = 1+np.arange(Nx)
g.es['label'] = g.es['capacity']
g.es['width'] = 10*g.es['capacity']
g.vs[[s, t]]['label'] = ["s", "t"]
g.vs[[s, t]]['x'] = [Nx/2-0.5, Nx/2-0.5]
g.vs[[s, t]]['y'] = [-1, 1]
g.vs[np.where(u0==0)[0]]["color"] = g.vs[t]['color'] =  0
g.vs[np.where(u0==1)[0]]["color"] = g.vs[s]['color'] = 255
display(ig.plot(g, **visual_style))
```

```{code-cell} ipython3
N = f.C.shape[0] - 2
f.pos = np.arange(N).tolist() + [N/2, N/2], np.zeros(N).tolist() + [N/2, -N/2]
ig, g, vs = f.plot()
ig.plot(g)
#display(p)
```

```{code-cell} ipython3
import igraph as ig
import numpy as np
from scipy.sparse import dok_matrix
u0 = np.array([0, 1, 1, 1, 0, 0, 1, 0])
Nx = len(u0)
C = dok_matrix((2+Nx,2+Nx))

s = source = Nx
t = target = Nx + 1

lam = 1.5

i0 = np.where(u0 == 0)[0]
i1 = np.where(u0 == 1)[0]
C[s, i1] = C[i0, t] = lam
for n in range(Nx-1):
    C[n, n+1] = 1
    C[n+1, n] = 1
#C += C.T
C = C.tocsr()

g = ig.Graph.Weighted_Adjacency(C, mode="directed", attr="capacity")
ig.plot(g)
```

```{code-cell} ipython3
from matplotlib import cm
import igraph as ig
import numpy as np
from scipy.sparse import dok_matrix
u0 = np.array([0, 1, 1, 1, 0, 0, 1, 0])
Nx = len(u0)
C = dok_matrix((2+Nx,2+Nx))

s = source = Nx
t = target = Nx + 1

lam = 1.5

i0 = np.where(u0 == 0)[0]
i1 = np.where(u0 == 1)[0]
C[s, i1] = C[i0, t] = lam
for n in range(Nx-1):
    C[n, n+1] = 1
    C[n+1, n] = 1
#C += C.T
C = C.tocsr()

g = ig.Graph.Weighted_Adjacency(C, mode="undirected", attr="capacity")
visual_style = dict(
    #edge_width=0.3,
    #vertex_size=1.5,
    palette="heat",
    bbox=(600,300),
    #layout="fruchterman_reingold"
)
g.vs['x'] = np.arange(Nx)
g.vs['y'] = 0
g.vs['label'] = 1+np.arange(Nx)
g.es['label'] = g.es['capacity']
g.es['width'] = 10*g.es['capacity']
g.vs[[s, t]]['label'] = ["s", "t"]
g.vs[[s, t]]['x'] = [Nx/2-0.5, Nx/2-0.5]
g.vs[[s, t]]['y'] = [-1, 1]
g.vs[np.where(u0==0)[0]]["color"] = g.vs[t]['color'] =  0
g.vs[np.where(u0==1)[0]]["color"] = g.vs[s]['color'] = 255
display(ig.plot(g, **visual_style))


res = g.maxflow(s, t, capacity=g.es["capacity"])
g.es['label'] = res.flow
g.es['width'] = 10*res.flow
g.es[flow.cut]['color'] = "red"
g.vs[flow.partition[0]]["color"] = 255
g.vs[flow.partition[1]]["color"] = 0
ig.plot(g, **visual_style)
```

```{code-cell} ipython3
print(f.max_flow_igraph().flow_value)
print(fn.max_flow_igraph().flow_value*n)
```

```{code-cell} ipython3
from math_583 import flow
import scipy.sparse
sp = scipy
np.random.seed(2)
for N in [10, 100, 1000]:
    for density in [0.01, 0.1, 0.5, 0.9]:
        print(f"{N=}, {density=}")
        C = (sp.sparse.random(N, N, density=0.01)*256).astype(int)
        f = flow.Flow(C)
        res_sp = f.max_flow_scipy()
        res_ig = f.max_flow_igraph()
        print(res_sp.flow_value, res_ig.flow_value)
        assert np.allclose(res_sp.flow_value, res_ig.flow_value)
        assert flow.allclose(res_sp.flow.C, res_ig.flow.C)
```

```{code-cell} ipython3
from math_583 import denoise

N = 256 // 2
Nxy = (N, N)
x, y = np.meshgrid(*[np.arange(_N) / _N for _N in Nxy], sparse=True)

Nc = 5
np.ogrid[1:Nc + 2, 1:Nc + 2]
_x, _y = np.ogrid[1:Nc + 1, 1:Nc + 1]
centers = np.ravel((_x + 1j * _y) / (Nc + 1))
radii = np.linspace(0, 1 / 2 / (Nc + 2), Nc**2 + 1)[1:]

circles = list(zip(centers, radii))

u = 0 * x + 0 * y
for z0, r in circles:
    u = np.maximum(u, np.where(abs(x + 1j * y - z0) < r, 1, 0))

scale = N / Nc / 8  # Why?
args = dict(sigma=0,
            lam=1 / scale,
            mode="wrap",
            p=1,
            q=1,
            eps_p=1e-6,
            eps_q=1e-6)


class MinCutMaxFlow(denoise.Base):
    u = None
    mode = "reflect"
    cval = 0.0

    def __init__(self, u=None, **kw):
        # Allow image to be provided as a positional argument.
        super().__init__(u=u, **kw)

    def init(self):
        # Ensure we have a characteristic function: only 0 or 1.
        u = self.u
        self.u = u.astype(bool)
        if not np.allclose(u, self.u):
            raise NotImplementedError(
                f"Data must be binary - 0 or 1 - got {set(u.ravel())}")
        if not len(u.shape) == 2:
            raise NotImplementedError(
                f"Only 2D images supported: got {u.shape=}")

    def compute_graph(self):
        """Compute the graph."""

    def pad(self, u=None, pad=2):
        """Return the extended u padded to satisfy the mode"""
        if u is None:
            u = self.u
        u = np.asarray(u)
        shape = tuple(_s + 2 * pad for _s in u.shape)
        u_pad = self.cval + np.zeros_like(u, shape=shape)
        u_pad[pad:-pad, pad:-pad] = u
        _i = slice(pad, -pad)
        _l = slice(pad, 2*pad)
        _r = slice(-2*pad, -pad)
        _ol = slice(0, pad)
        _or = slice(-pad,None)
        if self.mode == "constant":
            pass
        elif self.mode == "wrap":
            for u in u_pad, u_pad.T:
                u[_ol, _i] = u[_r, _i]
                u[_or, _i] = u[_l, _i]
                u[_ol, _or] = u[_r, _l]
            u = u_pad
            u[_ol, _ol] = u[_r, _r]
            u[_or:, _or] = u[_l, _l]
        elif self.mode == "reflect":
            for _u in u_pad, u_pad.T:
                _u[:pad, _i] = _u[2*pad-1:pad-1:-1, _i]
                _u[-pad:, _i] = _u[-pad-1:-2*pad-1:-1, _i]
        elif self.mode == "mirror":
            for _u in u_pad, u_pad.T:
                _u[:pad, :] = _u[-2 * pad:-pad, :]
                _u[-pad:, :] = _u[pad:2 * pad, :]
                
        return u_pad


x, y = np.meshgrid(1 + np.arange(3), 1 + np.arange(4), indexing='ij')
A = x + 10 * y
m = MinCutMaxFlow(u, mode="constant", cval=-1.0)
print(m.pad(A))

m = MinCutMaxFlow(u, mode="wrap")
print(m.pad(A))

m = MinCutMaxFlow(u, mode="reflect")
print(m.pad(A))
```

```{code-cell} ipython3
a = np.arange(5)
a[3:0:-1]
```

```{code-cell} ipython3
d = denoise.Denoise(im, **args)
u = d.minimize(plot=False, tol=1e-5)
clear_output()
im.show(u)
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
[min-cut max-flow theorem]: <https://en.wikipedia.org/wiki/Max-flow_min-cut_theorem>
[adjacency matrix]: <https://en.wikipedia.org/wiki/Adjacency_matrix>
[maximum flow problem]: <https://en.wikipedia.org/wiki/Maximum_flow_problem>
[minimum $s$-$t$ cut problem]: <https://en.wikipedia.org/wiki/Cut_(graph_theory)>
