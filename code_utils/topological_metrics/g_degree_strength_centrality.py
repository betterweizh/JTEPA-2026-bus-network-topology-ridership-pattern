import networkx as nx

from typing import Any, Dict, Hashable, Union, Tuple, TypeAlias

NodeMetric: TypeAlias = Dict[Hashable, Union[int, float]]


def _degree_undirected(
    G: nx.Graph,
    normalized: bool = False
) -> NodeMetric:
    """
    Return node degrees for an undirected graph.

    Parameters
    ----------
    G : nx.Graph
        Undirected NetworkX graph.
    normalized : bool, default False
        If True, returns degree centrality values in [0, 1]:
        degree / (n - 1). For n <= 1, returns 0.0 for all nodes.

    Returns
    -------
    dict
        {node: degree} if normalized=False
        {node: degree/(n-1)} if normalized=True
    """
    if G.is_directed():
        raise ValueError("G must be undirected.")

    if normalized:
        D = nx.degree_centrality(G)
    else:
        D = dict(G.degree())

    return D
# =============================================================
def _in_degree_directed(
    G: nx.DiGraph,
    normalized: bool = False
) -> NodeMetric:
    """
    In-degree for directed graphs.

    normalized=False -> {node: int in_degree}
    normalized=True  -> {node: float in_degree/(n-1)}  (in-degree centrality)
    """
    if not G.is_directed():
        raise ValueError("G must be directed.")

    if normalized:
        D = nx.in_degree_centrality(G)
    else:
        D = dict(G.in_degree())

    return D
# =============================================================
def _out_degree_directed(
    G: nx.DiGraph,
    normalized: bool = False
) -> NodeMetric:
    """
    Out-degree for directed graphs.

    normalized=False -> {node: int out_degree}
    normalized=True  -> {node: float out_degree/(n-1)}  (out-degree centrality)
    """
    if not G.is_directed():
        raise ValueError("G must be directed.")

    if normalized:
        D = nx.out_degree_centrality(G)
    else:
        D = dict(G.out_degree())

    return D
# =============================================================
def degree_centrality(
    G: Union[nx.Graph, nx.DiGraph],
    normalized: bool = False
) -> Union[
        NodeMetric,
        Tuple[NodeMetric, NodeMetric]
    ]:
    """
    Degree centrality for undirected or directed graphs.

    Returns:
    G undirected -> degree centrality
    G directed   -> (in-degree centrality, out-degree centrality)
    """
    if G.is_directed():
        iD = _in_degree_directed(G, normalized=normalized)
        oD = _out_degree_directed(G, normalized=normalized)
        return iD, oD
    else:
        D = _degree_undirected(G, normalized=normalized)
        return D
# =============================================================

def _strength_undirected(G, weight):
    """
    Strength (weighted degree) for undirected graphs.

    weight -> edge attribute name to use as weight
    returns -> {node: float strength}
    """
    if G.is_directed():
        raise ValueError("G must be undirected.")

    S = dict(G.degree(weight=weight))

    return S
# =============================================================
def _in_strength_directed(G, weight):
    """
    In-strength (weighted in-degree) for directed graphs.

    weight -> edge attribute name to use as weight
    returns -> {node: float in_strength}
    """
    if not G.is_directed():
        raise ValueError("G must be directed.")

    S = dict(G.in_degree(weight=weight))

    return S
# =============================================================
def _out_strength_directed(G, weight):
    """
    Out-strength (weighted out-degree) for directed graphs.

    weight -> edge attribute name to use as weight
    returns -> {node: float out_strength}
    """
    if not G.is_directed():
        raise ValueError("G must be directed.")

    S = dict(G.out_degree(weight=weight))

    return S
# =============================================================
def strength(
    G: Union[nx.Graph, nx.DiGraph],
    weight: str = None
) -> Union[
        NodeMetric,
        Tuple[NodeMetric, NodeMetric]
    ]:
    """
    Strength (weighted degree) for undirected or directed graphs.

    Returns:
    G undirected -> strength
    G directed   -> (in-strength, out-strength)
    """
    if G.is_directed():
        iS = _in_strength_directed(G, weight=weight)
        oS = _out_strength_directed(G, weight=weight)
        return iS, oS
    else:
        S = _strength_undirected(G, weight=weight)
        return S
# =============================================================