import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from segments.cluster import bootstrap_stability, compare_k, match_labels, ward_labels  # noqa: E402
from segments.features import (APR_JUN, BILLS, FEATURES, JUL_SEP, PAYMENTS, STATUS,  # noqa: E402
                               build_features)
from segments.over_time import carry_forward, outcome_by_transition, transitions  # noqa: E402
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


# Features for part of the six months

@pytest.fixture
def changed_behaviour():
    """Paid in full April to June, then fell behind July to September.

    Columns run from September back to April, and PAY_AMTn pays the bill in BILL_AMT(n+1).
    """
    return pd.DataFrame([customer(
        100_000,
        [2, 1, 0, -1, -1, -1],                              # Sep, Aug, Jul late or revolving; Jun to Apr paid
        [60_000, 60_000, 60_000, 10_000, 10_000, 10_000],   # bills Sep to Apr
        [0, 0, 1_000, 10_000, 10_000, 99_999],              # payments made Sep to Apr; the last pays March
    )])


def test_default_months_match_all_six(statements):
    assert build_features(statements).equals(build_features(statements, months=APR_JUN + JUL_SEP))


def test_periods_see_only_their_own_months(changed_behaviour):
    early = build_features(changed_behaviour, APR_JUN).loc[0]
    late = build_features(changed_behaviour, JUL_SEP).loc[0]
    assert early["share_paid_full"] == 1 and early["share_late"] == 0
    assert late["share_late"] == pytest.approx(2 / 3)
    assert late["share_revolving"] == pytest.approx(1 / 3)
    assert early["utilisation"] == pytest.approx(0.1)
    assert late["utilisation"] == pytest.approx(0.6)
    # Apr-Jun: the April and May bills (20,000) are paid in May and June (20,000).
    # The June bill is paid in July, outside the period, and the payment for March is ignored.
    assert early["payment_ratio"] == pytest.approx(1.0)
    # Jul-Sep: the July and August bills (120,000) are paid in August and September (0).
    assert late["payment_ratio"] == 0


def test_month_order_does_not_matter(statements):
    assert build_features(statements, ["Jun", "Apr", "May"]).equals(build_features(statements, APR_JUN))


def test_bad_months_are_rejected(statements):
    with pytest.raises(ValueError, match="unknown"):
        build_features(statements, ["Oct", "Sep"])
    with pytest.raises(ValueError, match="consecutive"):
        build_features(statements, ["Apr", "Jun"])


# Segments over time

def test_transitions_counts_and_rows():
    before = ["a", "a", "b", "b", "b"]
    after = ["a", "b", "b", "b", "a"]
    counts = transitions(before, after, order=["a", "b", "c"])
    assert counts.loc["a", "b"] == 1 and counts.loc["b", "b"] == 2
    assert counts.loc["c"].sum() == 0
    shares = transitions(before, after, normalize=True)
    assert np.allclose(shares.sum(axis=1), 1)
    assert shares.loc["b", "a"] == pytest.approx(1 / 3)


def test_outcome_by_transition():
    before = ["late", "late", "ok", "ok", "ok", "late"]
    after = ["late", "late", "late", "ok", "ok", "ok"]
    result = outcome_by_transition(before, after, [1, 0, 1, 0, 0, 1], into="late")
    assert result["customers"].to_dict() == {"stayed": 2, "moved in": 1, "moved out": 1, "never": 2}
    assert result.loc["stayed", "outcome_rate"] == 0.5
    assert result.loc["moved in", "outcome_rate"] == 1.0
    assert result.loc["never", "outcome_rate"] == 0.0


def test_carry_forward_on_unchanged_data_agrees(blobs):
    result = carry_forward(blobs, blobs, k=3)
    assert result.agreement == pytest.approx(1.0)
    assert (result.before == result.carried).all()
    assert (result.fresh == result.carried).all()


def test_carry_forward_follows_points_that_move(blobs):
    moved = blobs.copy()
    moved[:10] += [8, 0]                 # ten points from the first cluster move to the second
    result = carry_forward(blobs, moved, k=3)
    first, second = result.before[0], result.before[60]
    assert (result.carried[:10] == second).all()
    assert (result.carried[10:60] == first).all()
    assert result.agreement > 0.95
