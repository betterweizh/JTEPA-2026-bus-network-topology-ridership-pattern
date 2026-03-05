import numpy as np
import networkx as nx

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