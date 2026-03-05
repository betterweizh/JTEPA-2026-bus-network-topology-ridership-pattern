import copy
import math

import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.utils import check_array

from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.cluster import KMeans, AgglomerativeClustering

from typing import Any, Union, Optional, Sequence


def _delta_fast(mask_a, mask_b, distances):
    """
    Compute the minimum inter-cluster distance between two clusters.

    Parameters
    ----------
    mask_a : boolean np.array
        True for points belonging to cluster A
    mask_b : boolean np.array
        True for points belonging to cluster B
    distances : np.array [N, N]
        Pairwise distance matrix

    Returns
    -------
    float
        Minimum distance between any point in cluster A and any point in cluster B
    """
    # Extract distances between points in cluster A and cluster B
    vals = distances[mask_a][:, mask_b]

    # IMPORTANT: do NOT remove zeros here.
    # Zero can be a valid inter-cluster distance (identical points in different clusters).
    return np.min(vals)
# =============================================================================
def _big_delta_fast(mask, distances):
    """
    Compute the maximum intra-cluster distance (cluster diameter).

    Parameters
    ----------
    mask : boolean np.array
        True for points belonging to the cluster
    distances : np.array [N, N]
        Pairwise distance matrix

    Returns
    -------
    float
        Maximum distance between any two points within the cluster
    """
    # Extract distances among points in the same cluster
    vals = distances[mask][:, mask]

    # Diagonal values are zero and do not affect the maximum
    return np.max(vals)
# =============================================================================
def dunn_fast(points, labels):
    """
    Compute the Dunn Index.

    Dunn Index = (minimum inter-cluster distance) /
                 (maximum intra-cluster distance)

    Parameters
    ----------
    points : np.array [N, p]
        Data points
    labels : np.array [N]
        Cluster labels for each point

    Returns
    -------
    float
        Dunn index value
    """
    points = np.asarray(points, dtype=float)
    labels = np.asarray(labels, dtype=int)

    # Unique cluster labels
    ks = np.unique(labels)

    # Dunn index is undefined for fewer than 2 clusters
    if ks.size < 2:
        raise ValueError("Dunn index is undefined for fewer than 2 clusters.")

    # Precompute all pairwise distances once for efficiency
    distances = euclidean_distances(points)

    n_clusters = ks.size

    # Matrix of inter-cluster minimum distances
    # Diagonal is set to infinity so it will be ignored in min()
    deltas = np.full((n_clusters, n_clusters), np.inf, dtype=float)

    # Vector of intra-cluster maximum distances (cluster diameters)
    big_deltas = np.zeros(n_clusters, dtype=float)

    # Compute inter- and intra-cluster distances
    for i, ki in enumerate(ks):
        mask_i = (labels == ki)

        # Maximum distance within cluster i
        big_deltas[i] = _big_delta_fast(mask_i, distances)

        for j, kj in enumerate(ks):
            if i == j:
                continue

            mask_j = (labels == kj)

            # Minimum distance between cluster i and cluster j
            deltas[i, j] = _delta_fast(mask_i, mask_j, distances)

    # Smallest inter-cluster separation
    min_inter = float(np.min(deltas))

    # Largest cluster diameter
    max_intra = float(np.max(big_deltas))

    # Handle degenerate case:
    # If all clusters have zero diameter (e.g., singleton clusters),
    # avoid division by zero and define behavior explicitly.
    if max_intra == 0.:
        return np.inf if (min_inter > 0.) else 0.

    return min_inter / max_intra
# =============================================================================


def _validate_input(X : Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
    """
    Validate and coerce user-provided input into a clean 2D numeric NumPy array.

    This helper wraps scikit-learn's `check_array` with a strict configuration
    suitable for estimators/transformers that expect finite, numeric, dense data.

    Parameters
    ----------
    X : ArrayLike
        Input data to validate. Expected shape is (n_samples, n_features).
        Accepts array-like inputs such as:
        - numpy.ndarray
        - pandas.DataFrame / pandas.Series (will be converted to ndarray)

    Returns
    -------
    np.ndarray
        A validated 2D NumPy array with numeric dtype and finite values.
    """
    X = copy.deepcopy(X)

    X_checked = check_array(X,
        accept_sparse=False, accept_large_sparse=True, dtype='numeric',
        ensure_all_finite=True, ensure_2d=True, allow_nd=False)

    return X_checked
# ==========================================================================
def cluster_results(
    data: Union[np.ndarray, pd.DataFrame],
    n_clusters: int,
    standardization: bool = False,
    clustering_method: str = "kmeans",
    linkage: str = None,
) -> np.ndarray:
    """
    Fit a clustering model and return the assigned cluster labels.

    The function accepts a pandas DataFrame or any array-like input, optionally
    standardizes features, then fits either K-Means or hierarchical
    agglomerative clustering.

    Parameters
    ----------
    data : pandas.DataFrame or ArrayLike
        Input feature matrix where rows correspond to samples and columns to features.
        If a DataFrame is provided, its values are copied and used for clustering.
        If array-like, it is converted to a NumPy array.
    n_clusters : int
        The number of clusters to form. Must be >= 1.
    standardization : bool, default=False
        If True, standardize features to zero mean and unit variance using
        ``sklearn.preprocessing.StandardScaler`` before clustering.
    clustering_method : {"kmeans", "hierarchical"}, default="kmeans"
        The clustering algorithm to use:
        - "kmeans": uses ``sklearn.cluster.KMeans``
        - "hierarchical": uses ``sklearn.cluster.AgglomerativeClustering``
    linkage : {"ward", "complete", "average", "single"}, default="ward"
        Linkage criterion for hierarchical clustering. Only used when
        ``clustering_method="hierarchical"``.
        Notes:
        - If ``linkage="ward"``, the distance metric is Euclidean (scikit-learn constraint).
        - For other linkage methods, scikit-learn supports additional metrics depending
          on version and parameters.

    Returns
    -------
    input_array : numpy.ndarray
        The validated input data as a NumPy array of shape (n_samples, n_features).
        (This is the same data passed to the clustering model; if standardization=True,
        this returned array is still the *unstandardized* validated input.)
    labels : numpy.ndarray
        Integer cluster labels of shape (n_samples,).

    Raises
    ------
    ValueError
        If ``n_clusters`` is < 1, or if ``clustering_method`` is not recognized.
    """
    # Make a deep copy
    X = copy.deepcopy(data)
    # Validate and convert input
    X = _validate_input(X)

    # Standardization
    if standardization:
        X = StandardScaler().fit_transform(X)

    # clustering method
    method = clustering_method.lower()

    # Fit clustering model
    if method in {'kmeans'}:
        model = KMeans(
            n_clusters=n_clusters,
            random_state=42, tol=1.0e-9
        ).fit(X)

    elif method in {'hierarchical'}:
        model = AgglomerativeClustering(
            n_clusters = n_clusters,
            linkage = linkage
        ).fit(X)

    else:
        raise ValueError("clustering_method must be 'kmeans' or 'hierarchical'")

    # Assign labels
    input = np.asarray(X, dtype=float)
    label = np.asarray(model.labels_, dtype=int)

    return input, label
# ==================================================================================
def clustering_metric_search(
    data: Union[np.ndarray, pd.DataFrame],
    n_cluster_li: Optional[Sequence[int]] = None,
    standardization: bool = False,
    clustering_method = 'kmeans',
    linkage = None
):
    """
    Compute and plot multiple clustering validity indices for different numbers of clusters.

    Parameters
    ----------
    data : pd.DataFrame or np.ndarray
        Shape (n_samples, n_features). If DataFrame, values are used.
    max_clusters : int
        Evaluate cluster counts in range [2, max_clusters-1].
        (Matches your original loop.)
    standardization : bool
        If True, standardize features using StandardScaler before clustering/metrics.
    clustering_method : {'kmeans', 'hierarchical'}
        Clustering algorithm to use.
    linkage : str
        Linkage for AgglomerativeClustering when clustering_method='hierarchical'.

    Returns
    -------
    chi : pd.DataFrame
        DataFrame with index = number of clusters and columns = metric names.
    """
    n_samples = data.shape[0]

    # Prepare dict-of-dicts for results
    metric_dict = {
        'calinski_harabasz_score': {},
        'davies_bouldin_score': {},
        'silhouette_score': {},
        'dunn_score': {}
    }

    # Evaluate for k = 2..max_clusters-1 (your original behavior)
    for _n_clusters in tqdm.tqdm(n_cluster_li, desc="Evaluating clustering metrics"):

        # Guard against impossible settings
        if _n_clusters >= n_samples:
            # Many metrics (esp. silhouette) require n_clusters < n_samples
            continue

        # Some metrics can fail if a cluster is empty (rare here) or if labels are degenerate.
        for _metric, _func in zip(
            ['calinski_harabasz_score', 'davies_bouldin_score', 'silhouette_score', 'dunn_score'],
            [calinski_harabasz_score, davies_bouldin_score, silhouette_score, dunn_fast]
        ):

            try:
                # Compute cluster labels for this k
                input, labels = cluster_results(
                    data, _n_clusters,
                    standardization = standardization,
                    clustering_method = clustering_method,
                    linkage = linkage
                )
                metric_dict[_metric][_n_clusters] = _func(input, labels)

            except Exception:
                metric_dict[_metric][_n_clusters] = np.nan

    # Convert to DataFrame; index will be the cluster counts that were computed
    return pd.DataFrame(metric_dict).sort_index()
# =============================================================================




