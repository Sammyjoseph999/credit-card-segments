"""Turn six months of statements into a few behavioural features per customer."""
import numpy as np
import pandas as pd

STATUS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILLS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAYMENTS = [f"PAY_AMT{i}" for i in range(1, 7)]
TARGET = "default payment next month"

# Calendar month of each position in the column lists above (they run from September back to April).
MONTHS = ["Sep", "Aug", "Jul", "Jun", "May", "Apr"]
APR_JUN = ("Apr", "May", "Jun")
JUL_SEP = ("Jul", "Aug", "Sep")

FEATURES = ["log_limit", "utilisation", "share_inactive", "share_paid_full",
            "share_revolving", "share_late", "payment_ratio"]


def load(path):
    """Read the UCI file and give the columns readable names where it helps."""
    df = pd.read_csv(path)
    return df.rename(columns={TARGET: "default"})


def build_features(df, months=MONTHS):
    """Behavioural features, computed from limits, statements and repayment status only.

    `months` picks which of the six months (names from MONTHS) the features describe.
    The repayment status codes are -2 (no spending that month), -1 (paid in full),
    0 (revolving: paid at least the minimum) and 1 to 8 (months behind).
    Bills and payments are in columns ordered from the most recent month back,
    and PAY_AMTn is the payment made against the bill in BILL_AMT(n+1), in the following month.
    The payment ratio only uses bills whose payment also falls inside the chosen months,
    so features for one period never look at the next.
    """
    unknown = set(months) - set(MONTHS)
    if unknown:
        raise ValueError(f"unknown months: {sorted(unknown)}")
    cols = sorted(MONTHS.index(m) for m in set(months))
    paired = [i for i in cols if i - 1 in cols]
    if not paired:
        raise ValueError("need at least two consecutive months to compute the payment ratio")

    status = df[STATUS].to_numpy()[:, cols]
    bills = df[BILLS].to_numpy(float)
    payments = df[PAYMENTS].to_numpy(float)
    limit = df["LIMIT_BAL"].to_numpy(float)

    billed = bills[:, paired].clip(min=0).sum(axis=1)
    paid = payments[:, [i - 1 for i in paired]].sum(axis=1)
    # A customer with nothing to pay has, in effect, paid everything.
    payment_ratio = np.where(billed > 0, np.clip(paid / np.maximum(billed, 1), 0, 1.5), 1.0)

    return pd.DataFrame({
        "log_limit": np.log10(limit),
        "utilisation": np.clip(bills[:, cols].clip(min=0).mean(axis=1) / limit, 0, 1.2),
        "share_inactive": (status == -2).mean(axis=1),
        "share_paid_full": (status == -1).mean(axis=1),
        "share_revolving": (status == 0).mean(axis=1),
        "share_late": (status >= 1).mean(axis=1),
        "payment_ratio": payment_ratio,
    }, index=df.index)
