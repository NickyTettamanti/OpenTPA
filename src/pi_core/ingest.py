# src/pi_core/ingest.py
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict

STANDARD_COLS = [
    "claim_id","member_id","provider_id","service_date",
    "cpt_code","icd_code","geo","units",
    "billed_amount","allowed_amount","paid_amount","place_of_service",
]

# NOTE: keys are lowercase versions of your CSV headers.
DEFAULT_MAPPING: Dict[str, str] = {
    # your dataset (spaces, title case) -> standard
    "claim id": "claim_id",
    "provider id": "provider_id",
    "patient id": "member_id",
    "date of service": "service_date",
    "billed amount": "billed_amount",
    "procedure code": "cpt_code",
    "diagnosis code": "icd_code",
    "allowed amount": "allowed_amount",
    "paid amount": "paid_amount",
    # optional/other common aliases we may see
    "claimid":"claim_id","memberid":"member_id","providerid":"provider_id",
    "date_of_service":"service_date","service_date":"service_date",
    "cpt":"cpt_code","cpt_code":"cpt_code",
    "icd":"icd_code","icd_code":"icd_code",
    "state":"geo","zip3":"geo","geo":"geo",
    "qty":"units","units":"units",
    "billed":"billed_amount","allowed":"allowed_amount","paid":"paid_amount",
    "pos":"place_of_service","place_of_service":"place_of_service",
}

def _to_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # normalize headers to lowercase and map to our schema
    colmap: Dict[str,str] = {}
    for c in df.columns:
        lc = str(c).strip().lower()
        colmap[c] = DEFAULT_MAPPING.get(lc, lc)
    df = df.rename(columns=colmap)

    # ensure all standard columns exist
    for c in STANDARD_COLS:
        if c not in df.columns:
            df[c] = np.nan

    # parse dates (handles 8/7/2024)
    df["service_date"] = pd.to_datetime(df["service_date"], errors="coerce")

    # string-ish columns
    for c in ["claim_id","member_id","provider_id","cpt_code","icd_code","geo","place_of_service"]:
        df[c] = df[c].astype(str).str.strip()

    # numeric columns
    for c in ["units","billed_amount","allowed_amount","paid_amount"]:
        df[c] = df[c].apply(_to_float)

    # sensible defaults for fields your dataset doesn’t provide
    # allowed proxy if missing
    need_allowed = df["allowed_amount"].isna()
    df.loc[need_allowed, "allowed_amount"] = 0.8 * df.loc[need_allowed, "billed_amount"]

    # units default to 1 if missing/0
    df["units"] = df["units"].fillna(1.0).replace(0, 1.0)

    # geo not present -> use "UNK" so CPT×geo medians group correctly
    df["geo"] = df["geo"].replace({"nan": np.nan}).fillna("UNK")

    # place_of_service missing in your file -> mark as "NA"
    df["place_of_service"] = df["place_of_service"].replace({"nan": np.nan}).fillna("NA")

    # return in canonical column order
    return df[STANDARD_COLS].copy()
