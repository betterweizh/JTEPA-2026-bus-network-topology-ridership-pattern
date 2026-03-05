import numpy as np
import networkx as nx


def _graph_to_adjacency_matrix(G, weight=None, normalized=False):
    """
    Convert a NetworkX graph to a dense NumPy adjacency matrix.

    Parameters
    ----------
    G : networkx.Graph or networkx.DiGraph
        Input graph.
    weight : str or None
        Edge attribute name to use as weight. If None, returns an unweighted matrix
        (edges are represented by 1.0, non-edges by 0.0).
    normalized : bool
        If True, divide the matrix by its maximum entry (safe when max > 0).

    Returns
    -------
    M : (n, n) np.ndarray of float
        Adjacency/weight matrix in the node order list(G.nodes()).
    """
    # Build dense adjacency matrix in a consistent node order.
    M = nx.to_numpy_array(G, weight=weight, nodelist=list(G.nodes()), dtype=float)

    # Optional normalization (useful for weighted graphs to bound weights in [0,1]).
    if normalized:
        m_max = np.max(M)
        if m_max > 0:          # avoid division by zero for empty / zero-weight graphs
            M = M / m_max

    return M
# -----------------------------------------------------------------------------
def _clustering_coefficient_undirected(G, weight=None, normalized_mat=False):
    """
    Local clustering coefficient for UNDIRECTED graphs.

    Unweighted definition:
        C_i = 2*T_i / (k_i*(k_i-1))
    where:
        T_i = number of triangles that include node i
        k_i = degree of node i

    Matrix identity (simple undirected, no self-loops):
        diag(A^3)_i = 2*T_i  ->  T_i = diag(A^3)_i / 2

    Weighted variant here uses the common "intensity" approach:
        replace A by W^(1/3) in triangle counting.

    Returns
    -------
    C : np.ndarray shape (n,)
        Local clustering coefficients in the node order list(G.nodes()).
    """
    if nx.is_directed(G):
        raise ValueError("G must be undirected.")

    # Binary adjacency used for degree and triangle *structure*.
    # Even if the graph has weights, "unweighted adjacency" should be 0/1.
    A = _graph_to_adjacency_matrix(G, weight=None, normalized=normalized_mat)
    A = (A > 0).astype(float)

    # Degree for undirected graph: row-sum == col-sum
    k = A.sum(axis=1)

    if weight is None:
        # Unweighted triangle count: T_i = diag(A^3)/2
        T = np.diag(np.linalg.matrix_power(A, 3)) / 2.0
    else:
        # Weighted triangle "intensity":
        # Use W^(1/3) so that the product along a 3-cycle corresponds to geometric mean.
        W = _graph_to_adjacency_matrix(G, weight=weight, normalized=normalized_mat)
        W3 = np.power(W, 1.0 / 3.0)
        T = np.diag(np.linalg.matrix_power(W3, 3)) / 2.0

    # Allocate result, default 0 for nodes with degree < 2.
    C = np.zeros(A.shape[0], dtype=float)
    mask = k > 1
    C[mask] = (2.0 * T[mask]) / (k[mask] * (k[mask] - 1.0))

    return C
# -----------------------------------------------------------------------------
def _clustering_coefficient_directed_fagiolo(G, weight=None, normalized_mat=False):
    """
    Local clustering coefficient for DIRECTED graphs (Fagiolo 2007-style).

    This definition counts directed triangles by symmetrizing the adjacency:
        S = A + A^T
    Then:
        t_i = diag(S^3)_i / 2

    Denominator accounts for the number of possible triads around node i and
    corrects for reciprocal edges:
        denom_i = k_i^tot*(k_i^tot - 1) - 2*r_i
    where:
        k_i^tot = k_i^in + k_i^out
        r_i     = number of reciprocal edges incident to i
               = sum_j A_ij * A_ji

    Weighted variant uses W^(1/3) before symmetrization:
        S = W^(1/3) + (W^(1/3))^T

    Returns
    -------
    C : np.ndarray shape (n,)
        Local clustering coefficients in the node order list(G.nodes()).
    """
    if not nx.is_directed(G):
        raise ValueError("G must be directed.")

    # Binary adjacency for degrees and reciprocity correction.
    A = _graph_to_adjacency_matrix(G, weight=None, normalized=normalized_mat)
    A = (A > 0).astype(float)

    # Out-degree: sum across row (edges i -> j)
    out_deg = A.sum(axis=1)
    # In-degree: sum down column (edges j -> i)
    in_deg = A.sum(axis=0)

    # Total degree for directed case
    k_tot = in_deg + out_deg

    # Reciprocal edges per node i: count j such that i->j and j->i exist
    recip = np.sum(A * A.T, axis=1)

    if weight is None:
        # Symmetrize for triangle counting in directed setting
        S = A + A.T
        t = np.diag(np.linalg.matrix_power(S, 3)) / 2.0
    else:
        # Weighted symmetrized matrix per Fagiolo-style weighted extension
        W = _graph_to_adjacency_matrix(G, weight=weight, normalized=normalized_mat)
        W3 = np.power(W, 1.0 / 3.0)
        S = W3 + W3.T
        t = np.diag(np.linalg.matrix_power(S, 3)) / 2.0

    # Number of possible directed triads around node i (with reciprocity correction)
    denom = k_tot * (k_tot - 1.0) - 2.0 * recip

    # Compute clustering; define as 0 when denominator is 0 (no possible triads).
    C = np.zeros(A.shape[0], dtype=float)
    mask = denom > 0
    C[mask] = t[mask] / denom[mask]

    return C
# -----------------------------------------------------------------------------
def clustering_coefficient(G, weight=None, normalized_mat=False):
    """
    Dispatcher that returns local clustering coefficients for either
    directed or undirected graphs.

    Parameters
    ----------
    G : networkx.Graph or networkx.DiGraph
        Input graph.
    weight : str or None
        If None, compute the unweighted clustering coefficient.
        If a string, use that edge attribute as weight (weighted variant).
    normalized_mat : bool
        If True, normalize the adjacency/weight matrix by its max entry
        (only matters for weighted computations).

    Returns
    -------
    C : np.ndarray, shape (n,)
        Local clustering coefficients in the node order list(G.nodes()).
    """
    # Choose the correct definition based on graph type.
    # Directed: Fagiolo-style clustering (common directed generalization).
    # Undirected: standard local clustering coefficient.
    if nx.is_directed(G):
        C = _clustering_coefficient_directed_fagiolo(G, weight=weight, normalized_mat=normalized_mat)
    else:
        C = _clustering_coefficient_undirected(G, weight=weight, normalized_mat=normalized_mat)

    C = dict(zip(
        list(G.nodes()), [float(_val) for _val in C]
    ))

    return C
# ====================================================================================