"""Do segments built on one period still mean the same thing in the next?"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

from .cluster import fit_kmeans, match_labels


@dataclass
class CarryForward:
    before: np.ndarray   # segments fitted on the first period
    carried: np.ndarray  # second-period customers assigned with the first period's scaler and centroids
    fresh: np.ndarray    # a new fit on the second period, relabelled to line up with `carried`
    agreement: float     # adjusted Rand index between `carried` and `fresh`


def carry_forward(before, after, k=4, seed=0):
    """Fit k-means on `before`, assign `after` with the same scaler and centroids, and refit on `after`.

    `before` and `after` are feature tables for the same customers in two periods.
    If the segments hold, the carried-forward and fresh assignments agree closely.
    """
    scaler = StandardScaler().fit(before)
    model = fit_kmeans(scaler.transform(before), k, seed)
    carried = model.predict(scaler.transform(after))
    fresh = fit_kmeans(StandardScaler().fit_transform(after), k, seed).labels_
    return CarryForward(before=model.labels_, carried=carried,
                        fresh=match_labels(carried, fresh),
                        agreement=adjusted_rand_score(carried, fresh))


def transitions(before, after, order=None, normalize=False):
    """Customers by segment in the first period (rows) and the second (columns)."""
    table = pd.crosstab(pd.Series(np.asarray(before), name="before"),
                        pd.Series(np.asarray(after), name="after"))
    if order is not None:
        table = table.reindex(index=order, columns=order, fill_value=0)
    if normalize:
        table = table.div(table.sum(axis=1), axis=0)
    return table


def outcome_by_transition(before, after, outcome, into):
    """Customers and outcome rate for those who stayed in, moved into, left, or were never in `into`."""
    before, after = np.asarray(before), np.asarray(after)
    was, now = before == into, after == into
    group = np.select([was & now, ~was & now, was & ~now], ["stayed", "moved in", "moved out"], "never")
    result = pd.Series(np.asarray(outcome, float)).groupby(group).agg(["size", "mean"])
    result.columns = ["customers", "outcome_rate"]
    result = result.reindex(["stayed", "moved in", "moved out", "never"]).dropna()
    result["customers"] = result["customers"].astype(int)
    return result
