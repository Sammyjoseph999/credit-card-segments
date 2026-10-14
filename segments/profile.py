"""Describing segments in terms a lending team would use."""
import pandas as pd


def name_segment(row):
    """A plain-English name from a segment's average behaviour."""
    if row["share_inactive"] >= 0.5:
        return "Dormant"
    if row["share_late"] >= 0.4:
        return "Behind on payments"
    if row["share_paid_full"] + row["share_inactive"] >= 0.5 or row["payment_ratio"] >= 0.7:
        return "Pays in full"
    return "Revolving balance"


def profile(features, labels, outcome=None):
    """Mean feature values, size and (optionally) outcome rate per segment."""
    table = features.groupby(labels).mean()
    table.insert(0, "customers", features.groupby(labels).size())
    table.insert(1, "share", table["customers"] / table["customers"].sum())
    if outcome is not None:
        table["default_rate"] = pd.Series(outcome.to_numpy(), index=features.index).groupby(labels).mean()
    table.insert(0, "segment", table.apply(name_segment, axis=1))
    return table
