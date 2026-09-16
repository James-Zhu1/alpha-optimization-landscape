"""Map standardized alpha candidates into interpretable landscape views.

PCA provides a stable linear projection, UMAP emphasizes local neighborhoods,
and three clustering methods expose structure without treating any single
partition as ground truth.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.cluster import HDBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA


def _values(feature_matrix: pd.DataFrame | np.ndarray) -> np.ndarray:
    values = np.asarray(feature_matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("feature_matrix must have at least two rows")
    if not np.isfinite(values).all():
        raise ValueError("feature_matrix contains non-finite values")
    return values


def run_pca(
    feature_matrix: pd.DataFrame | np.ndarray,
    n_components: int = 3,
) -> NDArray[np.float64]:
    """Project a finite feature matrix onto its leading principal components."""
    if n_components < 1:
        raise ValueError("n_components must be positive")
    values = _values(feature_matrix)
    components = min(n_components, values.shape[0], values.shape[1])
    transformed = PCA(n_components=components, random_state=42).fit_transform(values)
    return np.asarray(transformed, dtype=np.float64)


def run_umap(
    feature_matrix: pd.DataFrame | np.ndarray,
    random_state: int = 42,
) -> NDArray[np.float64]:
    """Create a deterministic two-dimensional UMAP embedding."""
    os.environ.setdefault(
        "NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "alpha-landscape-numba")
    )
    try:
        import umap
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install umap-learn to generate the UMAP embedding") from exc
    values = _values(feature_matrix)
    neighbors = min(15, max(2, values.shape[0] - 1))
    transformed = umap.UMAP(
        n_components=2,
        n_neighbors=neighbors,
        min_dist=0.1,
        metric="euclidean",
        # Discrete expression features can create repeated/equivalent points;
        # random initialization avoids unstable spectral eigendecomposition.
        init="random",
        random_state=random_state,
    ).fit_transform(values)
    return np.asarray(transformed, dtype=np.float64)


def run_clustering(
    feature_matrix: pd.DataFrame | np.ndarray,
    n_clusters: int = 5,
) -> dict[str, np.ndarray]:
    """Cluster a feature matrix with the supported clustering algorithms."""
    if n_clusters < 2:
        raise ValueError("n_clusters must be at least 2")
    values = _values(feature_matrix)
    k = min(n_clusters, values.shape[0])
    labels = {
        "kmeans": KMeans(n_clusters=k, random_state=42, n_init=20).fit_predict(values),
        "agglomerative": AgglomerativeClustering(n_clusters=k).fit_predict(values),
    }
    # scikit-learn has shipped HDBSCAN since 1.3, so the density-based view needs
    # no compiled third-party package. That keeps the hosted dashboard installable
    # from wheels alone.
    min_cluster_size = min(max(5, values.shape[0] // 20), max(2, values.shape[0] - 1))
    labels["hdbscan"] = HDBSCAN(min_cluster_size=min_cluster_size).fit_predict(values)
    return labels
