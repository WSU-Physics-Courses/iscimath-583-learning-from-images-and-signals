from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse.csgraph

try:
    import igraph
except ImportError:
    igraph = None

try:
    import graph_tool.all
except ImportError:
    graph_tool = None

try:
    import networkx
except ImportError:
    networkx = None

from . import denoise

sp = scipy

__all__ = ["allclose", "Flow"]


def allclose(a, b, rtol=1e-5, atol=1e-8):
    """Return True if a and b are close.  Works on sparse matrices.

    https://stackoverflow.com/a/45135046/1088938
    """
    c = np.abs(np.abs(a - b) - rtol * np.abs(b))
    return c.max() <= atol


def prop_to_size(prop, mi=None, ma=5, log=False, power=0.5):
    """Convert prop to be more useful as a vertex size, or edge width.

    Similar behavior as :func:`graph_tool.draw.prop_to_size`
    """
    x = np.asarray(prop)
    if mi is None:
        if x.min() == 0:
            mi = 0.1
        else:
            mi = 1
    y = mi + (ma - mi) * ((x - x.min()) / (x.max() - x.min())) ** power
    return y


class Flow(denoise.Base):
    """Represents a flow network.

    This class provides an interface to various graph-theory libraries and useful tools
    for solving the min-cut max-flow problem, visualizing networks etc.

    Attributes
    ----------
    C : array-like
        Weighted adjacency matrix.  Usually, a sparse matrix whose entries represent the
        channel capacities (as an input problem) or the flows (as as a solution).
    s, t : int
        Index of the source (s) and target (t) vertices.
    """

    C = None
    s = 0
    t = -1
    pos = None

    def __init__(self, C, **kw):
        super().__init__(C=C, **kw)

    def init(self):
        super().init()

        # Convert s and t to positive integers
        N, _N = np.shape(self.C)
        assert N == _N
        self.s, self.t = [(_s + N) % N for _s in [self.s, self.t]]

    def plot(self, **kw):
        if graph_tool:
            return self.plot_graph_tool(**kw)
        if igraph:
            return self.plot_igraph(**kw)

    def plot_igraph(self, bbox=(600, 400), palette="heat", layout="circle"):
        """Plot the graph."""
        N = self.C.shape[0]
        g = igraph.Graph.Weighted_Adjacency(self.C, attr="capacity")
        visual_style = dict(
            # edge_width=0.3,
            # vertex_size=1.5,
            palette=palette,
            bbox=bbox,
            layout=layout,
        )
        if self.pos:
            g.vs["x"], g.vs["y"] = np.asarray(self.pos)
        g.vs["label"] = np.arange(N)
        g.vs[[self.s, self.t]]["label"] = ["s", "t"]

        capacity = np.asarray(g.es["capacity"])
        g.es["width"] = prop_to_size(capacity)
        g.es["label"] = np.where(capacity > 0, capacity, None)

        # g.vs[[self.s, self.t]]['x'] = [Nx/2-0.5, Nx/2-0.5]
        # g.vs[[self.s, self.t]]['y'] = [-1, 1]
        # g.vs[np.where(u0==0)[0]]["color"] = g.vs[t]['color'] =  0
        # g.vs[np.where(u0==1)[0]]["color"] = g.vs[s]['color'] = 255
        # g.vs['x'] = np.arange(Nx)
        return igraph.plot(g, **visual_style)

    def plot_graph_tool(self, pos=None, cut=None):
        # , bbox=(600, 400), palette="heat", layout="circle"):
        gt = graph_tool
        C = self.C
        g = gt.Graph(directed=not (allclose(C, C.T)))
        self._g = g
        idx = C.nonzero()
        weights = C[idx]
        g.add_edge_list(np.transpose(idx))
        ew = g.new_edge_property("float", weights)
        g.ep["edge_weight"] = ew
        if pos is None:
            if self.pos:
                pos = g.new_vertex_property("vector<double>")
                for vertex, xy in zip(g.vertices(), np.transpose(self.pos)):
                    pos[vertex] = xy
                pos[g.vertex(self.s)] = (0, -1)
                pos[g.vertex(self.t)] = (0, 1)

        vertex_text = list(g.vertex_index)
        vertex_text[self.s] = "s"
        vertex_text[self.t] = "t"
        vertex_text = g.new_vertex_property("string", vertex_text)

        visual_style = dict(
            edge_marker_size=10,
            edge_font_size=12,
            edge_text=ew,
            edge_pen_width=g.new_edge_property("double", prop_to_size(weights)),
            vertex_text=vertex_text,
            # vertex_size=100,
            # output_size=(2000, 600),
            # vertex_font_size=12,
        )
        if cut:
            S, T = cut.S, cut.T
            cut_edges = [
                _e for _e in g.edges() if _e.source() in S and _e.target() in T
            ]
            assert len(cut_edges) == len(cut.edges)
            visual_style["vertex_fill_color"] = g.new_vertex_property(
                "string", ["white" if _i in S else "yellow" for _i in g.vertex_index]
            )
            visual_style["edge_color"] = g.new_edge_property(
                "string",
                ["red" if _e in cut_edges else "black" for _e in g.edges()],
            )

            visual_style["edge_mid_marker"] = g.new_edge_property(
                "string",
                ["bar" if _e in cut_edges else "none" for _e in g.edges()],
            )

        return gt.all.graph_draw(
            g,
            pos=pos,
            **visual_style,
        )

    MaxFlowResult = namedtuple("MaxFlowResult", ["flow", "flow_value"])
    MinCutResult = namedtuple("MinCutResult", ["S", "T", "edges", "flow_value"])

    def max_flow(self, method="dinic"):
        """Return a (flow, flow_value) solving the maximal flow problem.

        Parameters
        ----------
        method: {'edmonds_karp', 'dinic', 'igraph', 'networkx'}
            The following methods are supported:

            * 'edmonds_karp', 'dinic': Use :func:`scipy.sparse.csgraph.maximum_flow`
            * 'igraph': Use :meth:`igraph.Graph.maxflow`
            * 'networkx': Use :func:`networkx.algorithms.flow.maximum_flow`

        Returns
        -------
        flow : Flow
            Flow network with flows.
        flow_value : float
            Maximal flow value.
        """
        if method in {"edmonds_karp", "dinic"}:
            return self.max_flow_scipy(method=method)
        if method == "igraph" and igraph:
            return self.max_flow_igraph()
        if method == "networkx" and networkx:
            return self.max_flow_networkx()
        raise ValueError(f"Unknown {method=} (or library could not be imported")

    def max_flow_scipy(self, method="dinic", factor=None):
        """Return a (flow, flow_value) solving the maximal flow problem.

        Uses :func:`scipy.sparse.csgraph.maximum_flow` which requires integer values.
        If self.C is not an integer, then we scale by factor to keep precision.

        Parameters
        ----------
        method : str
            See :func:`scipy.sparse.csgraph.maximum_flow`.
        factor : float, optional
            Factor to scale input by before converting to integers.  If not provided,
            then we use `np.iinfo(np.int64).max ** (1 / 2.1) / abs(C.max())`.  We have
            empirically determined this value, but it might not be safe for large
            matrices.


        Returns
        -------
        flow : Flow
            Flow network with flows. Note: scipy includes negative flows which are
            inconsistent with other tools.  We remove these.
        flow_value : float
            Maximal flow value.
        """
        C = sp.sparse.csr_matrix(self.C)

        if not C.dtype == np.int64:
            if factor is None:
                factor = np.iinfo(np.int64).max ** (1 / 2.1) / abs(C.max())
            C = np.round(factor * C, 0).astype(np.int64)
        else:
            factor = 1

        res = sp.sparse.csgraph.maximum_flow(
            C, source=self.s, sink=self.t, method=method
        )
        flow, flow_value = res.flow, res.flow_value

        if factor != 1:
            flow = flow / factor
            flow_value = flow_value / factor

        return self.MaxFlowResult(Flow(flow.maximum(0), s=self.s, t=self.t), flow_value)

    def max_flow_igraph(self):
        """Return a (flow, flow_value) solving the maximal flow problem.

        Uses :meth:`igraph.Graph.maxflow`.

        Returns
        -------
        flow : Flow
            Flow network with flows.
        flow_value : float
            Maximal flow value.
        """
        g = igraph.Graph.Weighted_Adjacency(self.C, attr="capacity")
        res = g.maxflow(self.s, self.t, capacity="capacity")
        self._res = res
        g.es["capacity"] = res.flow
        return self.MaxFlowResult(
            Flow(
                sp.sparse.csr_matrix(g.get_adjacency(attribute="capacity").data),
                s=self.s,
                t=self.t,
            ),
            res.value,
        )

    def min_cut(self):
        """Return a (S, T, edges, flow_value) solving the minimum cut flow problem.

        Uses :meth:`igraph.Graph.st_mincut`.

        Returns
        -------
        S, T : list of vertices
            Partition of the cut such that s ∈ S and t ∈ T.  The values are the indices
            of the vertices in the matrix C.
        edges : [(a, b)]
            Edges defining the cut with a ∈ S and b ∈ T.
        flow_value : float
            Maximal flow value.
        """
        return self.min_cut_igraph()

    def min_cut_igraph(self):
        """Return a (S, T, edges, flow_value) solving the minimum cut flow problem.

        Uses :meth:`igraph.Graph.st_mincut`.

        Returns
        -------
        S, T : list of vertices
            Partition of the cut such that s ∈ S and t ∈ T.  The values are the indices
            of the vertices in the matrix C.
        edges : [(a, b)]
            Edges defining the cut with a ∈ S and b ∈ T.
        flow_value : float
            Maximal flow value.
        """
        g = igraph.Graph.Weighted_Adjacency(self.C, attr="capacity")
        self._g = g
        res = g.st_mincut(self.s, self.t, capacity="capacity")
        S, T = res.partition
        edges = [_e.tuple for _e in g.es()[res.cut]]
        flow_value = res.value
        self._res = res
        return self.MinCutResult(S=S, T=T, edges=edges, flow_value=flow_value)

    def max_flow_networkx(self, method="dinitz"):
        """Return a (flow, flow_value) solving the maximal flow problem.

        Uses :func:`networkx.algorithms.flow.maximum_flow`.

        Returns
        -------
        flow : Flow
            Flow network with flows.
        flow_value : float
            Maximal flow value.
        """
        g = networkx.DiGraph(incoming_graph_data=sp.sparse.csr_matrix(self.C))
        flow_func = getattr(networkx.algorithms.flow.maxflow, method)
        R = flow_func(g, s=self.s, t=self.t, capacity="weight")
        return self.MaxFlowResult(
            Flow(
                networkx.adjacency_matrix(R, weight="flow").maximum(0),
                s=self.s,
                t=self.t,
            ),
            R.graph["flow_value"],
        )
