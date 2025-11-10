# src/pi_core/features.py
import numpy as np
import pandas as pd

def attach_reference(df: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    """
    Join CPT×geo reference medians and add derived features used by rules/ML.
    Falls back to CPT-only median if a geo-specific value is missing.
    """
    out = df.merge(ref, on=["cpt_code", "geo"], how="left")

    # Fallback: if CPT×geo missing, use global CPT median from the input claims
    if out["median_allowed"].isna().any():
        global_ref = (
            df.groupby("cpt_code")["allowed_amount"]
              .median()
              .rename("median_allowed_global")
              .reset_index()
        )
        out = out.merge(global_ref, on="cpt_code", how="left")
        out["median_allowed"] = out["median_allowed"].fillna(out["median_allowed_global"])
        out = out.drop(columns=["median_allowed_global"])

    # Derived features
    out["paid_to_billed"]  = np.where(out["billed_amount"] > 0, out["paid_amount"] / out["billed_amount"], np.nan)
    out["paid_to_allowed"] = np.where(out["allowed_amount"] > 0, out["paid_amount"] / out["allowed_amount"], np.nan)
    out["delta_vs_median"] = out["paid_amount"] - out["median_allowed"]
    out["log_paid"]        = np.log1p(out["paid_amount"].clip(lower=0))

    # Provider average paid (simple context feature)
    prov_avg = (
        out.groupby("provider_id")["paid_amount"]
          .mean()
          .rename("provider_avg_paid")
          .reset_index()
    )
    out = out.merge(prov_avg, on="provider_id", how="left")

    # CPT frequency rank (lower rank = more frequent)
    counts = out["cpt_code"].value_counts().rank(method="dense", ascending=False)
    out["cpt_freq_rank"] = out["cpt_code"].map(counts)

    return out
