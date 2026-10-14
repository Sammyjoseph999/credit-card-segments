import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from segments.cluster import bootstrap_stability, compare_k, match_labels, ward_labels  # noqa: E402
from segments.features import BILLS, FEATURES, PAYMENTS, STATUS, build_features  # noqa: E402
from segments.profile import name_segment, profile  # noqa: E402


def customer(limit, status, bills, payments):
    row = {"LIMIT_BAL": limit}
    row.update(dict(zip(STATUS, status)))
    row.update(dict(zip(BILLS, bills)))
    row.update(dict(zip(PAYMENTS, payments)))
    return row


@pytest.fixture
def statements():
    return pd.DataFrame([
        customer(100_000, [-1] * 6, [5_000] * 6, [5_000] * 6),           # pays in full
        customer(50_000, [0] * 6, [40_000] * 6, [2_000] * 6),            # revolving
        customer(200_000, [-2] * 6, [0] * 6, [0] * 6),                   # dormant
        customer(20_000, [2, 2, 1, 0, 0, 0], [21_000] * 6, [0] * 6),     # late
    ])


def test_features_have_expected_columns(statements):
    features = build_features(statements)
    assert list(features.columns) == FEATURES
    assert len(features) == 4


def test_status_shares_sum_to_one(statements):
    shares = build_features(statements)[["share_inactive", "share_paid_full",
                                          "share_revolving", "share_late"]]
    assert np.allclose(shares.sum(axis=1), 1)


def test_feature_values(statements):
    f = build_features(statements)
    assert f.loc[0, "payment_ratio"] == pytest.approx(1.0)
    assert f.loc[1, "payment_ratio"] == pytest.approx(0.05)
    assert f.loc[1, "utilisation"] == pytest.approx(0.8)
    assert f.loc[2, "payment_ratio"] == 1.0          # nothing billed counts as fully paid
    assert f.loc[3, "share_late"] == pytest.approx(0.5)
    assert f.loc[3, "utilisation"] == 1.05
    assert f.loc[0, "log_limit"] == pytest.approx(5.0)


def test_utilisation_is_capped():
    df = pd.DataFrame([customer(10_000, [0] * 6, [50_000] * 6, [0] * 6)])
    assert build_features(df).loc[0, "utilisation"] == 1.2


def test_segment_names(statements):
    f = build_features(statements)
    assert [name_segment(r) for _, r in f.iterrows()] == [
        "Pays in full", "Revolving balance", "Dormant", "Behind on payments"]


def test_profile_counts_and_outcome(statements):
    f = build_features(statements)
    table = profile(f, np.array([0, 0, 1, 1]), pd.Series([0, 1, 0, 1]))
    assert table["customers"].tolist() == [2, 2]
    assert table["share"].sum() == pytest.approx(1)
    assert table["default_rate"].tolist() == [0.5, 0.5]


def test_match_labels_undoes_a_permutation():
    reference = np.array([0, 0, 1, 1, 2, 2])
    assert match_labels(reference, np.array([2, 2, 0, 0, 1, 1])).tolist() == reference.tolist()


@pytest.fixture
def blobs():
    rng = np.random.default_rng(0)
    centres = np.array([[0, 0], [8, 0], [0, 8]])
    return np.vstack([c + rng.normal(size=(60, 2)) for c in centres])


def test_clear_clusters_are_found_and_stable(blobs):
    scores = compare_k(blobs, [2, 3, 4], sample_size=1000)
    assert scores["silhouette"].idxmax() == 3
    assert bootstrap_stability(blobs, 3, n_boot=5).min() > 0.95


def test_ward_agrees_with_truth(blobs):
    truth = np.repeat([0, 1, 2], 60)
    assert (match_labels(truth, ward_labels(blobs, 3)) == truth).mean() > 0.98
