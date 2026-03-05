import numpy as np
import pandas as pd
import networkx as nx

from typing import Optional, Hashable, Dict, Tuple


def select_katz_centrality_alpha(G, weight=None, safety=1.0):
    """
    Choose Katz centrality alpha automatically from the spectral radius ρ(A).

    A common sufficient condition for Katz convergence is:
        alpha < 1 / ρ(A)
    where ρ(A) = max_i |λ_i| is the spectral radius of the adjacency/weight matrix.

    This helper picks a *conservative*, human-friendly alpha as the largest power
    of 10 that does not exceed (safety / ρ(A)), with an extra guard to keep it
    strictly below 1/ρ(A).

    Parameters
    ----------
    G : networkx.Graph or networkx.DiGraph
        Input graph.
    weight : str or None, default None
        Edge attribute to use as weight. If None, treats edges as unweighted (1.0).
    safety : float, default 0.9
        Safety factor in (0, 1]. Values < 1 keep alpha strictly below the limit.
        If you set safety=1.0, we still enforce strict inequality via a small guard.

    Returns
    -------
    alpha : float
        Selected Katz alpha.
    rho : float
        Spectral radius of the adjacency/weight matrix.
    """
    if not (0 < safety <= 1.0):
        raise ValueError("safety must be in the interval (0, 1].")

    # Dense adjacency/weight matrix in the node order list(G.nodes()).
    # For large graphs, consider a sparse eigenvalue estimate instead.
    A = nx.to_numpy_array(G, weight=weight, dtype=float)

    # Spectral radius: works for directed graphs (complex eigvals) and signed weights.
    eigvals = np.linalg.eigvals(A)
    rho = float(np.max(np.abs(eigvals)))

    # Empty graph / all-zero weights => rho = 0, any alpha works.
    # Return a conventional value.
    if rho == 0.0:
        return 1.0, 0.0

    # Safe upper bound we target (may be equal to 1/rho if safety=1.0).
    limit = safety / rho
    if limit <= 0.0:
        # Should not happen with rho>0 and safety>0, but keep it safe.
        return 0.0, rho

    # Pick largest power of 10 <= limit.
    exp = np.floor(np.log10(limit))
    alpha = 10.0 ** exp

    # Ensure alpha <= limit (guard against floating error) and alpha < 1/rho strictly.
    # If safety=1.0, limit == 1/rho, so we still want alpha strictly below 1/rho.
    one_over_rho = 1.0 / rho
    while not (alpha <= limit and alpha < one_over_rho):
        alpha *= 0.1  # move to next smaller decade

    return float(alpha), rho
# ==============================================================================
def compute_directed_graph_topological_metrics(
    G: nx.DiGraph,
    weight: Optional[str] = None,
) -> pd.DataFrame:
    """
    Compute common topological / centrality metrics for a directed graph.

    This function computes several node-level metrics for a **directed** NetworkX graph
    and returns them as a DataFrame (metrics as rows, nodes as columns).

    Parameters
    ----------
    G : nx.DiGraph
        A directed NetworkX graph. Must satisfy ``G.is_directed() == True``.
        Node labels may be any hashable type.
    weight : str or None, default=None
        Edge attribute name to use as weight/distance, depending on the metric:
        - For eigenvector centrality, Katz centrality, betweenness, clustering, and PageRank:
          interpreted as edge weight (larger typically means "stronger connection").
        - For closeness centrality:
          passed as ``distance=weight`` (larger means "farther"/more costly).
        If None, all edges are treated as weight 1.

    Returns
    -------
    pandas.DataFrame
        DataFrame with:
        - index: metric names (row-wise)
        - columns: graph nodes
        - values: metric value for each node

        The DataFrame index name is set to "metric".

    Raises
    ------
    ValueError
        If ``G`` is not directed.

    Notes
    -----
    - "in_*" metrics are computed on ``G``.
    - "out_*" metrics are computed by reversing the graph (``G.reverse(copy=False)``),
      so that "out" behavior can be expressed as "in" behavior on the reversed graph.
    - This function expects a helper named ``select_katz_centrality_alpha`` to be
      available in scope. It should return a tuple ``(alpha, something_else)``.
    - Some metrics (especially eigenvector centrality) may fail to converge for
      certain graphs; NetworkX may raise an exception in that case.

    Metrics computed
    ----------------
    - in_degree_centrality / out_degree_centrality (actually raw degrees, not normalized)
    - in_eigenvector_centrality / out_eigenvector_centrality
    - in_katz_centrality / out_katz_centrality
    - closeness_centrality
    - betweenness_centrality
    - clustering_coefficient
    - pagerank
    """
    if not G.is_directed():
        raise ValueError("G must be directed.")

    # Reverse view of the graph (no copy) to compute "out" analogs as "in" on reversed.
    Gr = G.reverse(copy=False)

    # Container for metric name -> {node: value}
    indices: Dict[str, Dict[Hashable, float]] = {}

    # Degree Centrality
    # NOTE: these are raw degrees, not normalized "degree centrality"
    indices['in_degree_centrality'] = dict(G.in_degree())
    indices['out_degree_centrality'] = dict(G.out_degree())

    # Eigenvector Centrality
    indices['in_eigenvector_centrality'] = nx.eigenvector_centrality(
        G, weight = weight,
        tol=1.0e-6, max_iter=1000)

    indices['out_eigenvector_centrality'] = nx.eigenvector_centrality(
        Gr, weight = weight,
        tol=1.0e-6, max_iter=1000)

    # Katz Centrality
    alpha, _ = select_katz_centrality_alpha(G, weight=weight)
    indices['in_katz_centrality'] = nx.katz_centrality(
        G, alpha = alpha, weight = weight)

    alpha, _ = select_katz_centrality_alpha(Gr, weight=weight)
    indices['out_katz_centrality'] = nx.katz_centrality(
        Gr, alpha = alpha, weight = weight)

    # Closeness Centrality
    indices['closeness_centrality'] = nx.closeness_centrality(
        G, distance = weight)

    # Betweenness Centrality
    indices['betweenness_centrality'] = nx.betweenness_centrality(
        G, weight = weight, normalized = True)

    # Clustering Coefficient
    indices['clustering_coefficient'] = nx.clustering(
        G, weight = weight)

    # PageRank
    indices['pagerank'] = nx.pagerank(
        G, weight = weight,
        alpha = 0.85, tol=1e-7, max_iter=1000)

    # Build a metric-by-node DataFrame
    indices = pd.DataFrame.from_dict(indices, orient='columns')
    indices.columns.name = "metric"

    return indices
# =============================================================================
def compute_undirected_graph_topological_metrics(
    G: nx.Graph,
    weight: Optional[str] = None,
) -> pd.DataFrame:
    """
    Compute common topological / centrality metrics for an undirected graph.

    This function computes several node-level metrics for an **undirected** NetworkX graph
    and returns them as a DataFrame (metrics as rows, nodes as columns).

    Parameters
    ----------
    G : nx.Graph
        An undirected NetworkX graph. Must satisfy ``G.is_directed() == False``.
        Node labels may be any hashable type.
    weight : str or None, default=None
        Edge attribute name to use as weight/distance, depending on the metric:
        - For eigenvector centrality, Katz centrality, betweenness, clustering, and PageRank:
          interpreted as edge weight (larger typically means "stronger connection").
        - For closeness centrality:
          passed as ``distance=weight`` (larger means "farther"/more costly).
        If None, all edges are treated as weight 1.

    Returns
    -------
    pandas.DataFrame
        DataFrame with:
        - index: metric names (row-wise)
        - columns: graph nodes
        - values: metric value for each node

        The DataFrame index name is set to "metric".

    Raises
    ------
    ValueError
        If ``G`` is directed.

    Notes
    -----
    - This function expects a helper named ``select_katz_centrality_alpha`` to be
      available in scope. It should return a tuple ``(alpha, something_else)``.
    - Some metrics (especially eigenvector centrality) may fail to converge for
      certain graphs; NetworkX may raise an exception in that case.

    Metrics computed
    ----------------
    - degree_centrality
    - eigenvector_centrality
    - katz_centrality
    - closeness_centrality
    - betweenness_centrality
    - clustering_coefficient
    - pagerank: Undirected graphs will be converted to a directed graph
        with two directed edges for each undirected edge.
    """
    if G.is_directed():
        raise ValueError("G must be undirected.")

    # Container for metric name -> {node: value}
    indices: Dict[str, Dict[Hashable, float]] = {}

    # Degree Centrality
    # NOTE: this is raw degree, not normalized "degree centrality"
    indices['degree_centrality'] = dict(G.degree())

    # Eigenvector Centrality
    indices['eigenvector_centrality'] = nx.eigenvector_centrality(
        G, weight = weight,
        tol=1.0e-6, max_iter=1000)

    # Katz Centrality
    alpha, _ = select_katz_centrality_alpha(G, weight=weight)
    indices['katz_centrality'] = nx.katz_centrality(
        G, alpha = alpha, weight = weight)

    # Closeness Centrality
    indices['closeness_centrality'] = nx.closeness_centrality(
        G, distance = weight)

    # Betweenness Centrality
    indices['betweenness_centrality'] = nx.betweenness_centrality(
        G, weight = weight, normalized = True)

    # Clustering Coefficient
    indices['clustering_coefficient'] = nx.clustering(
        G, weight = weight)

    # PageRank
    indices['pagerank'] = nx.pagerank(
        G, weight = weight,
        alpha = 0.85, tol=1e-7, max_iter=1000)

    # Build a metric-by-node DataFrame
    indices = pd.DataFrame.from_dict(indices, orient='columns')
    indices.columns.name = "metric"

    return indices
# =============================================================================