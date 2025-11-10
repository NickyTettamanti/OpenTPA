# src/pi_core/refprices.py
import pandas as pd

def build_ref_prices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns median allowed by CPT × geo.
    Falls back to CPT-only median handled later in features.attach_reference.
    """
    ref = (
        df.groupby(["cpt_code", "geo"], dropna=False)["allowed_amount"]
          .median()
          .reset_index()
          .rename(columns={"allowed_amount": "median_allowed"})
    )
    ref["median_allowed"] = ref["median_allowed"].round(2)
    return ref
