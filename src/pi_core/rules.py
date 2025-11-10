# src/pi_core/rules.py
import pandas as pd
import numpy as np


def _empty_flags() -> pd.DataFrame:
    """Return an empty flags DataFrame with the canonical columns.

    The rule runner concatenates parts and only requires at minimum
    claim_id, flag_type, reason. Keeping this consistent simplifies
    downstream code.
    """
    return pd.DataFrame(columns=["claim_id", "flag_type", "reason"])


def _ensure_columns(df: pd.DataFrame, cols):
    """Ensure dataframe has the listed columns (fill with NaN if missing)."""
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    return df

# thresholds
OVERPAY_MULT = 1.05        # paid > 105% of allowed
PRICE_OUTLIER_MULT = 1.5   # paid > 1.5 × median_allowed

# new rule thresholds
LOW_UNITS_MAX = 2          # small number of units considered 'low'
LOW_UNITS_MULT = 3.0       # paid-per-unit > 3× median-per-unit
MEDICARE_MULT = 1.10       # paid significantly above median for Medicare

def rule_duplicate(df: pd.DataFrame) -> pd.DataFrame:
    key = ["member_id", "provider_id", "service_date", "cpt_code"]
    # require the keys for duplication check
    if not all(k in df.columns for k in key + ["claim_id"]):
        return _empty_flags()

    mask = df.duplicated(subset=key, keep=False)
    out = df.loc[mask, ["claim_id"] + key].copy()
    out["flag_type"] = "RULE_DUPLICATE"
    out["reason"] = "Duplicate: same member/provider/date/CPT"
    return out

def rule_overpay(df: pd.DataFrame) -> pd.DataFrame:
    m = OVERPAY_MULT
    if not all(c in df.columns for c in ["claim_id", "paid_amount", "allowed_amount"]):
        return _empty_flags()

    # coerce numerics
    df = _ensure_columns(df, ["paid_amount", "allowed_amount"])
    df["paid_amount"] = pd.to_numeric(df["paid_amount"], errors="coerce").fillna(0.0)
    df["allowed_amount"] = pd.to_numeric(df["allowed_amount"], errors="coerce").fillna(0.0)

    mask = df["paid_amount"] > m * df["allowed_amount"]
    out = df.loc[mask, ["claim_id", "paid_amount", "allowed_amount"]].copy()
    out["flag_type"] = "RULE_OVERPAY"
    out["reason"] = out.apply(
        lambda r: f"Over-payment: paid {r.paid_amount:.2f} > {m:.2f}×allowed {r.allowed_amount:.2f}", axis=1
    )
    return out

def rule_price_outlier(df: pd.DataFrame) -> pd.DataFrame:
    m = PRICE_OUTLIER_MULT
    if not all(c in df.columns for c in ["claim_id", "paid_amount", "median_allowed"]):
        return _empty_flags()

    df = _ensure_columns(df, ["paid_amount", "median_allowed"])
    df["paid_amount"] = pd.to_numeric(df["paid_amount"], errors="coerce").fillna(0.0)
    df["median_allowed"] = pd.to_numeric(df["median_allowed"], errors="coerce").fillna(0.0)

    mask = df["paid_amount"] > m * df["median_allowed"]
    out = df.loc[mask, ["claim_id", "paid_amount", "median_allowed", "cpt_code", "geo"]].copy()
    out["flag_type"] = "RULE_PRICE_OUTLIER"
    out["reason"] = out.apply(
        lambda r: f"Price outlier: paid {r.paid_amount:.2f} > {m:.2f}×median {r.median_allowed:.2f} (CPT {r.cpt_code}, {r.geo})",
        axis=1
    )
    return out

def rule_low_units_high_charge(df: pd.DataFrame) -> pd.DataFrame:
    """Flag claims with very few units but unusually high per-unit charge.

    - units <= LOW_UNITS_MAX
    - (paid_amount / units) > LOW_UNITS_MULT * (median_allowed / units)
    """
    if not all(c in df.columns for c in ["claim_id", "units", "paid_amount", "median_allowed"]):
        return _empty_flags()

    # guard against zero units and coerce numerics
    df = _ensure_columns(df, ["units", "paid_amount", "median_allowed"])
    df["units"] = pd.to_numeric(df["units"], errors="coerce").fillna(1.0).replace(0, 1.0)
    df["paid_amount"] = pd.to_numeric(df["paid_amount"], errors="coerce").fillna(0.0)
    df["median_allowed"] = pd.to_numeric(df["median_allowed"], errors="coerce").fillna(0.0)

    units = df["units"]
    paid_per_unit = df["paid_amount"] / units
    median_per_unit = df["median_allowed"] / units
    mask = (units <= LOW_UNITS_MAX) & (paid_per_unit > LOW_UNITS_MULT * median_per_unit)
    out = df.loc[mask, ["claim_id", "units", "paid_amount", "median_allowed", "cpt_code", "geo"]].copy()
    out["flag_type"] = "RULE_LOW_UNITS_HIGH_CHARGE"
    out["reason"] = out.apply(
        lambda r: (
            f"Low units ({int(r.units)}) but high unit price: paid/unit {r.paid_amount/r.units:.2f} > "
            f"{LOW_UNITS_MULT:.1f}×median/unit {r.median_allowed/r.units:.2f} (CPT {r.cpt_code})"
        ), axis=1
    )
    return out

def rule_medicare_rate_variance(df: pd.DataFrame) -> pd.DataFrame:
    """Flag potential Medicare rate variance where Medicare claims are paid materially above the median.

    Conditions (demo heuristic):
    - insurer indicates Medicare (case-insensitive match)
    - paid_amount > MEDICARE_MULT * median_allowed OR paid_to_allowed > MEDICARE_MULT
    """
    if "claim_id" not in df.columns:
        return _empty_flags()

    df = _ensure_columns(df, ["insurance_type", "paid_amount", "median_allowed", "paid_to_allowed"])
    payer_mask = df["insurance_type"].fillna("").str.contains("medicare", case=False, na=False)
    df["paid_amount"] = pd.to_numeric(df["paid_amount"], errors="coerce").fillna(0.0)
    df["median_allowed"] = pd.to_numeric(df["median_allowed"], errors="coerce").fillna(0.0)
    df["paid_to_allowed"] = pd.to_numeric(df["paid_to_allowed"], errors="coerce").fillna(0.0)

    ratio_mask = (df["paid_amount"] > MEDICARE_MULT * df["median_allowed"]) | (df["paid_to_allowed"] > MEDICARE_MULT)
    mask = payer_mask & ratio_mask
    out = df.loc[mask, ["claim_id", "insurance_type", "paid_amount", "median_allowed", "paid_to_allowed", "cpt_code"]].copy()
    out["flag_type"] = "RULE_MEDICARE_VARIANCE"
    out["reason"] = out.apply(
        lambda r: f"Medicare variance: paid {r.paid_amount:.2f} vs median {r.median_allowed:.2f} (paid/allowed {getattr(r,'paid_to_allowed',None)})", axis=1
    )
    return out

def run_rules(df: pd.DataFrame) -> pd.DataFrame:
    """Run all deterministic checks and return a flags DataFrame with claim_id, flag_type, reason."""
    parts = [
        rule_duplicate(df),
        rule_overpay(df),
        rule_price_outlier(df),
        rule_low_units_high_charge(df),
        rule_medicare_rate_variance(df),
    ]
    out = pd.concat(parts, ignore_index=True)
    return out.drop_duplicates(["claim_id","flag_type"])
