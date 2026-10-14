"""Choosing the number of segments and checking that they are stable."""
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, silhouette_score


def fit_kmeans(X, k, seed=0):
    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)


def compare_k(X, ks, sample_size=6000, seed=0):
    """Inertia, silhouette and Davies-Bouldin for each k.

    Silhouette is quadratic in the number of points, so it is scored on a fixed sample.
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), min(sample_size, len(X)), replace=False)
    rows = []
    for k in ks:
        model = fit_kmeans(X, k, seed)
        rows.append({"k": k, "inertia": model.inertia_,
                     "silhouette": silhouette_score(X[idx], model.labels_[idx]),
                     "davies_bouldin": davies_bouldin_score(X, model.labels_)})
    return pd.DataFrame(rows).set_index("k")


def bootstrap_stability(X, k, n_boot=20, seed=0):
    """Agreement (adjusted Rand index) between a reference fit and fits on bootstrap resamples.

    Each resampled model labels the full data, so the labels are compared on the same points.
    A value near 1 means the segments come out the same however the customers are sampled.
    """
    reference = fit_kmeans(X, k, seed).labels_
    rng = np.random.default_rng(seed)
    scores = []
    for b in range(n_boot):
        idx = rng.integers(0, len(X), len(X))
        model = fit_kmeans(X[idx], k, seed + b + 1)
        scores.append(adjusted_rand_score(reference, model.predict(X)))
    return np.array(scores)


def ward_labels(X, k):
    """Agglomerative clustering with Ward linkage, labels 0 to k-1."""
    return fcluster(linkage(X, method="ward"), t=k, criterion="maxclust") - 1


def match_labels(reference, other):
    """Relabel `other` so that its clusters line up with `reference` as closely as possible."""
    k = max(reference.max(), other.max()) + 1
    counts = np.zeros((k, k), dtype=int)
    np.add.at(counts, (other, reference), 1)
    rows, cols = linear_sum_assignment(-counts)
    mapping = dict(zip(rows, cols))
    return np.array([mapping[label] for label in other])
