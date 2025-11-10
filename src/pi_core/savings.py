# src/pi_core/savings.py
import pandas as pd
import numpy as np

def rule_savings(df: pd.DataFrame, flags: pd.DataFrame) -> pd.DataFrame:
    """
    Conservative, rule-only savings per claim.
    - RULE_OVERPAY: paid - allowed (>= 0)
    - RULE_PRICE_OUTLIER: paid - median_allowed (>= 0)
    - RULE_DUPLICATE: treat paid as recoverable (max-by-claim later avoids double-counting)

    Uses *_base columns from the merged base table to avoid conflicts with any
    similarly named columns that might appear in `flags`.
    """
    base = df[["claim_id","paid_amount","allowed_amount","median_allowed"]].drop_duplicates("claim_id")

    # Merge with explicit suffix so we always know which columns to use
    f = flags.merge(base, on="claim_id", how="left", suffixes=("", "_base"))

    # Ensure numeric and no NaNs for math
    for c in ["paid_amount_base", "allowed_amount_base", "median_allowed_base"]:
        if c not in f.columns:
            f[c] = np.nan
        f[c] = pd.to_numeric(f[c], errors="coerce").fillna(0.0)

    def _sv(row) -> float:
        if row["flag_type"] == "RULE_OVERPAY":
            return max(row["paid_amount_base"] - row["allowed_amount_base"], 0.0)
        if row["flag_type"] == "RULE_PRICE_OUTLIER":
            return max(row["paid_amount_base"] - row["median_allowed_base"], 0.0)
        if row["flag_type"] == "RULE_DUPLICATE":
            return float(row["paid_amount_base"])
        return 0.0

    f["est_savings"] = f.apply(_sv, axis=1)

    # Per-claim max avoids double counting across rule types
    out = f.groupby("claim_id", as_index=False)["est_savings"].max()
    return out
