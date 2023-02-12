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
```

(sec:FlatNorm)=
Flat Norm
=========

Here we consider a faster approach to minimizing $E[u]$ in the case where $u$ is a
characteristic function $u \in \{0, 1\}$:

\begin{gather*}
  E[u] = \int \abs{\vect{\nabla}u} + \lambda \int \abs{u-d}.
\end{gather*}

The algorithm (see {cite:p}`Vixie:2010` for details).

```{code-cell} ipython3
C = np.random.random((5,5))
import scipy.sparse.csgraph
import scipy as sp
```

```{code-cell} ipython3
from math_583 import denoise

u0 = []
N = 6
for n in range(N):
    u0.extend([0]*n)
    u0.extend([1]*n)

u0 = np.asarray(u0)

im = denoise.Image(u0)
lam = 2./(N/2)
d = denoise.Denoise(im, sigma=0, p=1, q=1, lam=lam, eps_p=1e-8, eps_q=1e-8, mode="constant")
u = d.minimize(np.asarray(u0)*0+1, callback=None)

fig, ax = plt.subplots()
ax.plot(u0, label="Original")
ax.plot(u, label=f"2/λ={2/d.lam:.4}")
ax.legend();
```

```{code-cell} ipython3
from scipy.sparse import dok_matrix
from scipy.sparse.csgraph import maximum_flow

Nx = len(u0)
C = dok_matrix((2+Nx,2+Nx), dtype=int)

source = Nx
sink = Nx + 1

def toint(x, factor=2**16):
    return int(np.round(x*factor))

assert lam <= 1

i0 = np.where(u0 == 0)[0]
i1 = np.where(u0 == 1)[0]
C[source, i1] = C[i0, sink] = toint(lam)
for n in range(Nx-1):
    C[n, n+1] = toint(1)
    C[n+1, n] = toint(1)
C += C.T
C = C.tocsr()
res = maximum_flow(C, source=source, sink=sink)
flow = getattr(res, 'flow', res.residual)
#print(flow)
```

```{code-cell} ipython3
import networkx as nx
from packaging import version
assert version.parse("3.0.0") <= version.parse(nx.__version__)


G = nx.DiGraph(incoming_graph_data=C.tocsr())
f = 10
pos = {
    n:(f*n, 0) for n in range(Nx)
}
pos[source] = (f*Nx/2, f*Nx/2)
pos[sink] = (f*Nx/2, -f*Nx/2)
nx.draw(G, pos=pos)
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

```{code-cell} ipython3
import numpy as np
import graph_tool.all
import graph_tool as gt
from scipy.sparse import dok_matrix
from scipy.sparse.csgraph import maximum_flow

u0 = np.array([0, 1, 1, 1, 0, 0, 1, 0])
Nx = len(u0)
C = dok_matrix((2+Nx,2+Nx))

source = Nx
sink = Nx + 1

lam = 1.5

i0 = np.where(u0 == 0)[0]
i1 = np.where(u0 == 1)[0]
C[source, i1] = C[i0, sink] = lam
for n in range(Nx-1):
    C[n, n+1] = 1
    C[n+1, n] = 1
#C += C.T
C = C.tocsr()

g = gt.Graph(directed=True)
idx = C.nonzero()
weights = C[idx]
g.add_edge_list(np.transpose(idx))
ew = g.new_edge_property("float")
ew.a = weights
g.ep['edge_weight'] = ew

res = gt.all.boykov_kolmogorov_max_flow(g, g.vertex(source), g.vertex(sink), ew)
#res = gt.all.edmonds_karp_max_flow(g, g.vertex(source), g.vertex(sink), ew)
part = gt.all.min_st_cut(g, g.vertex(source), ew, res)
f = 10
pos = g.new_vertex_property("vector<double>")
for n in range(Nx):
    pos[g.vertex(n)] = (f*n, 0)
pos[g.vertex(sink)] = (f*Nx/2, f*Nx/4)
pos[g.vertex(source)] = (f*Nx/2, -f*Nx/4)
gt.all.graph_draw(g, pos=pos, 
                  edge_pen_width=gt.all.prop_to_size(res, mi=0, ma=5, power=1),
                  edge_text=res,
                  vertex_fill_color=part, 
                  vertex_text=g.vertex_index,
                  #vertex_size=100,
                  output_size=(2000, 600),
                  #edge_marker_size=10,
                  #vertex_font_size=12, 
                  edge_font_size=12)
```

```{code-cell} ipython3
g.new_vertex_property("string")
```

# [igraph][]

[igraph]: https://igraph.org/

```{code-cell} ipython3
try: 
    import igraph as ig
except ImportError:
    import sys
    !{sys.executable} -m pip install --user python-igraph
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
res.flow
```

```{code-cell} ipython3
import sys
import scipy.sparse
sp = scipy
np.shape([[1,2,3]])
```

```{code-cell} ipython3
m = g.get_adjacency(attribute="capacity")
```

```{code-cell} ipython3
from importlib import reload
from math_583 import flow
reload(flow)
import scipy.sparse
sp = scipy

C = np.array([[0, 16, 13,  0,  0,  0],
              [0,  0, 10, 12,  0,  0],
              [0,  4,  0,  0, 14,  0],
              [0,  0,  9,  0,  0, 20],
              [0,  0,  0,  7,  0,  4],
              [0,  0,  0,  0,  0,  0]])
f = flow.Flow(C, s=0, t=5)
print(f.max_flow_scipy().flow.C.toarray())
print(f.max_flow_igraph().flow.C.toarray())
print(f.max_flow_networkx(method='edmonds_karp').flow.C.toarray())
print(f.max_flow_networkx(method='dinitz').flow.C.toarray())
print(f.max_flow_networkx(method='preflow_push').flow.C.toarray())
print(f.max_flow_networkx(method='boykov_kolmogorov').flow.C.toarray())
print(f.max_flow_networkx(method='shortest_augmenting_path').flow.C.toarray())
```

```{code-cell} ipython3
#C = [[0, 2], 
#     [1, 0]]
f = flow.Flow(C, s=0, t=5)
for res in [f.max_flow_scipy(), f.max_flow_igraph()]:
    print(res.flow_value)
    print(res.flow.C.toarray())
```

```{code-cell} ipython3
f._res.flow
```

```{code-cell} ipython3
import scipy.sparse.csgraph
sp = scipy
graph = sp.sparse.csr_matrix([[0, 5, 0], [0, 0, 3], [0, 0, 0]])
res = sp.sparse.csgraph.maximum_flow(graph, 0, 2)
print(res.flow_value)
print(res.flow.toarray())
```

```{code-cell} ipython3
res.flow.maximum(0).toarray()
```

```{code-cell} ipython3
import networkx as nx
C = np.array([[0, 16, 13,  0,  0,  0],
              [0,  0, 10, 12,  0,  0],
              [0,  4,  0,  0, 14,  0],
              [0,  0,  9,  0,  0, 20],
              [0,  0,  0,  7,  0,  4],
              [0,  0,  0,  0,  0,  0]])

g = nx.DiGraph(C)
res = nx.maximum_flow(g, _s=0, _t=5, capacity='weight')
res = nx.algorithms.flow.dinitz(g, s=0, t=5, capacity='weight')
res = nx.algorithms.flow.boykov_kolmogorov(g, s=0, t=5, capacity='weight')
nx.adjacency_matrix(res, weight='flow').maximum(0).toarray()
res.graph['flow_value']
```

```{code-cell} ipython3
flow = res[1]
#for e in flow:
f = nx.DiGraph()
f.add_weighted_edges_from(flow)
```

```{code-cell} ipython3

```

```{code-cell} ipython3

```
