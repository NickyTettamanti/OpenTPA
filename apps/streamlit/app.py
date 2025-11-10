"""OpenTPA Payment Integrity Streamlit app (recreated).

Pages:
- Home: project description and choose data file (upload or select from data/raw)
- Overview: high-level KPIs and flag summary table
- Filter & Triage: browse flagged claims, filter, select, export
- Selected Claims: view and summary of currently selected claims
- Recovery Tracker: queue selected claims and track progress

This is a minimal, self-contained implementation intended to be easy to extend.
"""

import os
import sys
from typing import List

import pandas as pd
import math
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder, DataReturnMode, GridUpdateMode
from st_aggrid.shared import JsCode
import json
import io
from datetime import datetime

# make local src importable for pi_core modules
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import pi_core.ingest as ingest
import pi_core.refprices as refprices
import pi_core.features as features
import pi_core.rules as rules
import pi_core.savings as savings
from recoup_utils import load_tracker, save_tracker, upsert_tracker, provider_contact, write_letter_file, render_letter_text


st.set_page_config(page_title="OpenTPA | Payment Integrity", layout="wide")


def compute_pipeline(df_raw: pd.DataFrame):
    df_std = ingest.standardize_columns(df_raw)
    # normalize claim_id to a trimmed string to avoid type/whitespace mismatches
    if "claim_id" in df_std.columns:
        df_std["claim_id"] = df_std["claim_id"].astype(str).str.strip()
    ref_path = os.path.join("data", "ref", "ref_prices.csv")
    if os.path.exists(ref_path):
        ref = pd.read_csv(ref_path)
        source_label = "ref_prices.csv"
    else:
        ref = refprices.build_ref_prices(df_std)
        source_label = "computed medians"
    df_feat = features.attach_reference(df_std, ref)
    # ensure canonical claim_id in feature set
    if "claim_id" in df_feat.columns:
        df_feat["claim_id"] = df_feat["claim_id"].astype(str).str.strip()
    flags = rules.run_rules(df_feat)
    sav = savings.rule_savings(df_feat, flags)
    # ensure est_savings exists
    sav = sav.assign(est_savings=lambda d: d.get("est_savings", 0.0).fillna(0.0))
    return df_feat, flags, sav, source_label


def _pipeline_cache_key():
    """Create a lightweight cache key for the staged claims used to decide whether to rerun the pipeline.

    Uses (active_file name, row count, unique claim ids, file mtime when available). This is conservative
    but avoids expensive hashing of entire DataFrames.
    """
    df_raw = st.session_state.get("claims_raw_df")
    active = st.session_state.get("active_file", "") or ""
    rows = len(df_raw) if df_raw is not None else 0
    unique = int(df_raw["claim_id"].nunique()) if (df_raw is not None and "claim_id" in df_raw.columns) else 0
    mtime = None
    try:
        if active and os.path.isdir(os.path.join(os.getcwd(), "data", "raw")):
            path = os.path.join(os.getcwd(), "data", "raw", active)
            if os.path.exists(path):
                mtime = os.path.getmtime(path)
    except Exception:
        mtime = None
    return f"{active}|{rows}|{unique}|{mtime}"


def _format_money_columns(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Return a copy of df where specified cols are formatted as dollar strings with floor to whole dollars."""
    if df is None or df.empty:
        return df
    out = df.copy()
    for c in cols:
        if c in out.columns:
            try:
                out[c] = out[c].fillna(0).apply(lambda v: f"${int(math.floor(float(v))):,}")
            except Exception:
                # best-effort: coerce to string
                out[c] = out[c].apply(lambda v: f"${str(v)}")
    return out


def _build_fiduciary_excel(sel_df: pd.DataFrame, prov_agg: pd.DataFrame, reason_agg: pd.DataFrame, ml_df: pd.DataFrame = None) -> bytes:
    """Build an in-memory Excel file with three sheets: Selected Claims, Providers, Reason Groups.

    Monetary columns are floored to whole dollars (integers) for consistency with UI display.
    Returns bytes of the .xlsx file. Requires openpyxl (pandas engine).
    """
    try:
        import openpyxl  # noqa: F401
    except Exception:
        raise ImportError("openpyxl is required to build Excel reports")

    out = io.BytesIO()
    # prepare copies with floored monetary columns
    sel_copy = sel_df.copy()
    for c in ("allowed_amount", "paid_amount", "est_savings"):
        if c in sel_copy.columns:
            try:
                sel_copy[c] = sel_copy[c].fillna(0).apply(lambda v: int(math.floor(float(v or 0))))
            except Exception:
                pass

    prov_copy = prov_agg.copy()
    for c in ("total_allowed", "total_paid", "recoverable_amount"):
        if c in prov_copy.columns:
            try:
                prov_copy[c] = prov_copy[c].fillna(0).apply(lambda v: int(math.floor(float(v or 0))))
            except Exception:
                pass

    reason_copy = reason_agg.copy()
    if "recoverable" in reason_copy.columns:
        try:
            reason_copy["recoverable"] = reason_copy["recoverable"].fillna(0).apply(lambda v: int(math.floor(float(v or 0))))
        except Exception:
            pass

    # write sheets
    with pd.ExcelWriter(out, engine="openpyxl") as xw:
        try:
            sel_copy.to_excel(xw, sheet_name="Selected Claims", index=False)
            prov_copy.to_excel(xw, sheet_name="Providers", index=False)
            reason_copy.to_excel(xw, sheet_name="Reason Groups", index=False)
            # optional ML-only candidates sheet
            if ml_df is not None and not ml_df.empty:
                ml_copy = ml_df.copy()
                # floor monetary columns if present
                for c in ("allowed_amount", "paid_amount", "est_savings"):
                    if c in ml_copy.columns:
                        try:
                            ml_copy[c] = ml_copy[c].fillna(0).apply(lambda v: int(math.floor(float(v or 0))))
                        except Exception:
                            pass
                ml_copy.to_excel(xw, sheet_name="ML Candidates", index=False)
        except Exception:
            # if any sheet write fails, raise
            raise
    out.seek(0)
    return out.read()


def compute_ml_anomalies(df_feat: pd.DataFrame, pct: float = 0.02) -> pd.DataFrame:
    """Run an IsolationForest unsupervised anomaly detector on sensible numeric features.

    Returns a DataFrame with columns: claim_id, ml_score (decision_function), ml_recoup_candidate (bool)
    - pct controls the contamination (fraction flagged as anomalies). If scikit-learn is not installed,
      raises ImportError.
    """
    try:
        from sklearn.ensemble import IsolationForest
    except Exception as e:
        raise ImportError("scikit-learn is required for ML anomaly detection") from e

    if df_feat is None or df_feat.empty:
        return pd.DataFrame(columns=["claim_id", "ml_score", "ml_recoup_candidate"])

    # pick numeric columns that are likely meaningful for anomalies
    candidate_cols = [
        c
        for c in ["paid_amount", "allowed_amount", "est_savings", "paid_to_allowed", "units"]
        if c in df_feat.columns
    ]
    if not candidate_cols:
        return pd.DataFrame({"claim_id": df_feat.get("claim_id", pd.Series(dtype=str)).astype(str).tolist(),
                             "ml_score": [float('nan')] * len(df_feat),
                             "ml_recoup_candidate": [False] * len(df_feat)})

    X = df_feat[candidate_cols].copy()
    # coerce numerics and fill missing
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0.0)

    try:
        iso = IsolationForest(random_state=42, n_estimators=100, contamination=max(0.0001, float(pct)))
        iso.fit(X)
        scores = iso.decision_function(X)  # higher => more normal
        preds = iso.predict(X)  # -1 anomaly, 1 normal
    except Exception:
        # if the model fails for any reason, return empty-ish frame
        return pd.DataFrame({"claim_id": df_feat.get("claim_id", pd.Series(dtype=str)).astype(str).tolist(),
                             "ml_score": [float('nan')] * len(df_feat),
                             "ml_recoup_candidate": [False] * len(df_feat)})

    out = pd.DataFrame({"claim_id": df_feat["claim_id"].astype(str).tolist(),
                        "ml_score": scores.tolist(),
                        "ml_recoup_candidate": (pd.Series(preds) == -1).tolist()})
    return out


def get_pipeline_results():
    """Return cached pipeline outputs (df_feat, flags, sav, source_label).

    Caches results in `st.session_state['pipeline_cache']` keyed by _pipeline_cache_key().
    Also stores an aggregated flags table in that cache as 'flags_agg'.
    """
    df_raw = st.session_state.get("claims_raw_df")
    if df_raw is None or df_raw.empty:
        return None, None, None, None
    key = _pipeline_cache_key()
    existing_key = st.session_state.get("pipeline_cache_key")
    cache = st.session_state.get("pipeline_cache", None)
    if existing_key == key and cache is not None:
        return cache.get("df_feat"), cache.get("flags"), cache.get("sav"), cache.get("source_label")
    # compute and store (call the real pipeline implementation)
    df_feat, flags, sav, source_label = compute_pipeline(df_raw)
    try:
        flags_agg = aggregate_flags(flags)
    except Exception:
        flags_agg = aggregate_flags(pd.DataFrame())
    st.session_state["pipeline_cache"] = {"df_feat": df_feat, "flags": flags, "sav": sav, "flags_agg": flags_agg, "source_label": source_label}
    st.session_state["pipeline_cache_key"] = key
    return df_feat, flags, sav, source_label


def _prune_session_selected_to_features(df_feat: pd.DataFrame):
    """Prune `st.session_state['selected_claims']` to only IDs present in df_feat.

    Keeps ordering of the existing session selection but removes IDs not in the processed features.
    """
    try:
        if df_feat is None or df_feat.empty:
            st.session_state["selected_claims"] = []
            try:
                _write_selected_debug_file([])
            except Exception:
                pass
            return
        valid = set(df_feat["claim_id"].astype(str).str.strip().unique())
        current = st.session_state.get("selected_claims", [])
        normalized = [str(x).strip() for x in current if x is not None]
        pruned = [x for x in normalized if x in valid]
        # preserve order and dedupe
        pruned_final = list(dict.fromkeys(pruned))
        st.session_state["selected_claims"] = pruned_final
        try:
            _write_selected_debug_file(pruned_final)
        except Exception:
            pass
    except Exception:
        # best-effort; don't crash the app on pruning
        pass


def aggregate_flags(flags: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-claim flags into a single-row-per-claim dataframe.

    Produces columns:
      - claim_id
      - flag_types: comma-separated sorted unique flag_type values
      - reason: first reason value (keeps a representative reason)

    This avoids duplicating feature rows when merging flags (many claims can have >1 flag row).
    """
    if flags is None or flags.empty:
        return pd.DataFrame(columns=["claim_id", "flag_types", "reason"])
    try:
        f = flags.copy()
        # normalize to string to avoid type issues
        if "flag_type" in f.columns:
            f["flag_type"] = f["flag_type"].astype(str).replace({"nan": ""})
        else:
            f["flag_type"] = ""
        if "reason" not in f.columns:
            f["reason"] = ""

        def _join_flags(s):
            vals = [str(x).strip() for x in s if pd.notna(x) and str(x).strip() not in ("", "nan")]
            return ",".join(sorted(set(vals)))

        agg = (
            f.groupby("claim_id", dropna=False)
            .agg(flag_types=("flag_type", _join_flags), reason=("reason", "first"))
            .reset_index()
        )
        return agg
    except Exception:
        # best-effort: fall back to a minimal mapping
        try:
            return flags[["claim_id"]].drop_duplicates().assign(flag_types="", reason="")
        except Exception:
            return pd.DataFrame(columns=["claim_id", "flag_types", "reason"])


def reason_group_from_flag(flag_type: str) -> str:
    """Map granular flag_type to one of four higher-level reason groups.

    Groups chosen:
    - Duplicate
    - Overpayment
    - Price/Unit
    - Insurance variance
    """
    if not isinstance(flag_type, str):
        return "Other"
    m = flag_type.upper()
    if m == "RULE_DUPLICATE":
        return "Duplicate"
    if m == "RULE_OVERPAY":
        return "Overpayment"
    if m in ("RULE_PRICE_OUTLIER", "RULE_LOW_UNITS_HIGH_CHARGE"):
        return "Price/Unit"
    if m == "RULE_MEDICARE_VARIANCE":
        return "Insurance variance"
    return "Other"


def load_claims_from_raw(choice: str = None) -> pd.DataFrame:
    # list csv files in data/raw
    raw_dir = os.path.join(os.getcwd(), "data", "raw")
    files = []
    if os.path.isdir(raw_dir):
        files = [f for f in os.listdir(raw_dir) if f.lower().endswith(".csv")]
    # if user picked a file name, load it
    if choice and choice in files:
        return pd.read_csv(os.path.join(raw_dir, choice))
    # fallback to sample if exists
    demo = os.path.join(raw_dir, "claims_sample.csv")
    if os.path.exists(demo):
        return pd.read_csv(demo)
    return pd.DataFrame()


def ensure_session_state_keys():
    st.session_state.setdefault("claims_raw_df", None)
    st.session_state.setdefault("selected_claims", [])
    st.session_state.setdefault("active_file", None)


ensure_session_state_keys()


def _build_fiduciary_docx(sel_df: pd.DataFrame, prov_agg: pd.DataFrame, reason_agg: pd.DataFrame, meta: dict, rule_agg: pd.DataFrame = None, logo_bytes: bytes = None) -> bytes:
    """Build a simple Word (.docx) fiduciary report containing the provided tables.

    Returns bytes for download. If python-docx isn't installed, raises ImportError.
    """
    try:
        from docx import Document
        from docx.shared import Pt, Inches
    except Exception as e:
        raise ImportError("python-docx is required to build .docx reports") from e

    doc = Document()
    doc.styles['Normal'].font.name = 'Calibri'
    doc.styles['Normal'].font.size = Pt(11)

    # optional logo
    if logo_bytes:
        try:
            bio_img = io.BytesIO(logo_bytes)
            doc.add_picture(bio_img, width=Inches(1.5))
        except Exception:
            # ignore logo failures
            pass

    # title and client
    doc.add_heading(meta.get('title', 'Fiduciary Report'), level=1)
    if meta.get('client_name'):
        doc.add_paragraph(f"Client: {meta.get('client_name')} ({meta.get('client_id','')})")
    doc.add_paragraph(f"Generated: {meta.get('generated_at', '')}")
    doc.add_paragraph(f"Claims staged: {meta.get('n_staged', 0)}")
    doc.add_paragraph(f"Selected claims: {meta.get('n_selected', 0)}")
    doc.add_paragraph(f"Total potential recoverable: ${meta.get('total_recoverable', 0.0):,.2f}")

    def _add_table_from_df(df: pd.DataFrame, title: str, max_rows: int = None):
        doc.add_heading(title, level=2)
        if df is None or df.empty:
            doc.add_paragraph("No rows")
            return
        rows = df.shape[0]
        cols = df.shape[1]
        if max_rows is not None:
            data = df.head(max_rows)
        else:
            data = df
        table = doc.add_table(rows=1, cols=cols)
        hdr_cells = table.rows[0].cells
        for i, h in enumerate(data.columns.tolist()):
            hdr_cells[i].text = str(h)
        for _, r in data.iterrows():
            row_cells = table.add_row().cells
            for i, v in enumerate(r.tolist()):
                row_cells[i].text = str(v)

    # Data provenance and metadata
    doc.add_heading("Data provenance", level=2)
    doc.add_paragraph(f"Source file: {meta.get('data_file', '')}")
    doc.add_paragraph(f"Source label: {meta.get('source_label', '')}")
    doc.add_paragraph(f"Analysis period: {meta.get('analysis_period', '')}")
    doc.add_paragraph(f"Pipeline key: {meta.get('pipeline_key', '')}")

    # Methodology and assumptions
    doc.add_heading("Methodology & assumptions", level=2)
    doc.add_paragraph(meta.get('methodology', 'Rules-based flagging and recoverable estimate using configured rule set. See rules documentation for details.'))

    # Rules applied summary
    if rule_agg is not None and not rule_agg.empty:
        _add_table_from_df(rule_agg, "Rules applied (counts)")

    # include the full selected claims table
    _add_table_from_df(sel_df, "Selected claims (full)")
    _add_table_from_df(prov_agg, "Provider aggregates")
    _add_table_from_df(reason_agg, "Top reason groups")

    # Attestation / signature block (populate from meta if available)
    doc.add_heading("Attestation", level=2)
    signer_line = meta.get('signer_name') or "Signer: ____________________________"
    title_line = meta.get('signer_title') or "Title: ____________________________"
    date_line = meta.get('signature_date') or "Date: ____________________________"
    doc.add_paragraph(meta.get('attestation_text', "I attest that the data and analysis in this report were prepared following standard procedures and the rules defined in the Rules Applied section."))
    doc.add_paragraph(f"Signer: {signer_line}")
    doc.add_paragraph(f"Title: {title_line}")
    doc.add_paragraph(f"Date: {date_line}")

    # Disclaimer
    doc.add_heading("Disclaimer", level=2)
    doc.add_paragraph("This report is provided for fiduciary oversight purposes. Estimates of recoverable amounts are based on rule-based algorithms and should be validated before any recoupment actions. The provider/plan should follow contractual and legal processes for recovery.")

    bio = io.BytesIO()
    doc.save(bio)
    bio.seek(0)
    return bio.read()


def _build_fiduciary_pdf(sel_df: pd.DataFrame, prov_agg: pd.DataFrame, reason_agg: pd.DataFrame, meta: dict, rule_agg: pd.DataFrame = None, logo_bytes: bytes = None) -> bytes:
    """Build a simple PDF fiduciary report. Requires reportlab. Returns bytes.
    If reportlab isn't available, raises ImportError.
    """
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, Spacer
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet
    except Exception as e:
        raise ImportError("reportlab is required to build PDF reports") from e

    bio = io.BytesIO()
    doc = SimpleDocTemplate(bio, pagesize=letter)
    styles = getSampleStyleSheet()
    elems = []
    # optional logo
    if logo_bytes:
        try:
            from reportlab.platypus import Image as RLImage
            bio_img = io.BytesIO(logo_bytes)
            elems.append(RLImage(bio_img, width=100, height=40))
        except Exception:
            pass

    elems.append(Paragraph(meta.get('title', 'Fiduciary Report'), styles['Title']))
    if meta.get('client_name'):
        elems.append(Paragraph(f"Client: {meta.get('client_name')} ({meta.get('client_id','')})", styles['Normal']))
    elems.append(Paragraph(f"Generated: {meta.get('generated_at', '')}", styles['Normal']))
    elems.append(Paragraph(f"Claims staged: {meta.get('n_staged', 0)}", styles['Normal']))
    elems.append(Paragraph(f"Selected claims: {meta.get('n_selected', 0)}", styles['Normal']))
    elems.append(Paragraph(f"Total potential recoverable: ${meta.get('total_recoverable', 0.0):,.2f}", styles['Normal']))
    elems.append(Spacer(1, 12))

    # Data provenance
    elems.append(Paragraph('Data provenance', styles['Heading2']))
    elems.append(Paragraph(f"Source file: {meta.get('data_file', '')}", styles['Normal']))
    elems.append(Paragraph(f"Source label: {meta.get('source_label', '')}", styles['Normal']))
    elems.append(Paragraph(f"Analysis period: {meta.get('analysis_period', '')}", styles['Normal']))
    elems.append(Paragraph(f"Pipeline key: {meta.get('pipeline_key', '')}", styles['Normal']))
    elems.append(Spacer(1, 12))

    elems.append(Paragraph('Methodology & assumptions', styles['Heading2']))
    elems.append(Paragraph(meta.get('methodology', 'Rules-based flagging and recoverable estimate using configured rule set. See rules documentation for details.'), styles['Normal']))
    elems.append(Spacer(1, 12))

    # Rules applied summary
    if rule_agg is not None and not rule_agg.empty:
        _table_from_df(rule_agg, 'Rules applied (counts)')

    def _table_from_df(df: pd.DataFrame, title: str, max_rows: int = 200):
        elems.append(Paragraph(title, styles['Heading2']))
        if df is None or df.empty:
            elems.append(Paragraph("No rows", styles['Normal']))
            return
        data = [list(df.columns)]
        for _, r in df.head(max_rows).iterrows():
            data.append([str(x) for x in r.tolist()])
        t = Table(data, repeatRows=1)
        t.setStyle([
            ('GRID', (0,0), (-1,-1), 0.25, colors.grey),
            ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
        ])
        elems.append(t)
        elems.append(Spacer(1, 12))

    _table_from_df(sel_df, 'Selected claims (first 200 rows)')
    _table_from_df(prov_agg, 'Provider aggregates')
    _table_from_df(reason_agg, 'Top reason groups')

    elems.append(Spacer(1, 12))
    elems.append(Paragraph('Attestation', styles['Heading2']))
    elems.append(Paragraph('I attest that the data and analysis in this report were prepared following standard procedures and the rules defined in the Rules Applied section.', styles['Normal']))
    elems.append(Paragraph('Signer: ____________________________', styles['Normal']))
    elems.append(Paragraph('Title: ____________________________', styles['Normal']))
    elems.append(Paragraph('Date: ____________________________', styles['Normal']))
    elems.append(Spacer(1, 12))
    elems.append(Paragraph('Disclaimer', styles['Heading2']))
    elems.append(Paragraph('This report is provided for fiduciary oversight purposes. Estimates of recoverable amounts are based on rule-based algorithms and should be validated before any recoupment actions. The provider/plan should follow contractual and legal processes for recovery.', styles['Normal']))

    # Attestation block (populate from meta if available)
    elems.append(Spacer(1, 12))
    elems.append(Paragraph('Attestation', styles['Heading2']))
    signer_line = meta.get('signer_name') or '____________________________'
    title_line = meta.get('signer_title') or '____________________________'
    date_line = meta.get('signature_date') or '____________________________'
    elems.append(Paragraph(meta.get('attestation_text', 'I attest that the data and analysis in this report were prepared following standard procedures and the rules defined in the Rules Applied section.'), styles['Normal']))
    elems.append(Paragraph(f'Signer: {signer_line}', styles['Normal']))
    elems.append(Paragraph(f'Title: {title_line}', styles['Normal']))
    elems.append(Paragraph(f'Date: {date_line}', styles['Normal']))

    doc.build(elems)
    bio.seek(0)
    return bio.read()


def _write_selected_debug_file(ids: List[str]):
    """Write a small debug file containing the current selected_claims list and a timestamp.

    This is used to capture the session selection from the running app for offline diagnostics.
    """
    try:
        os.makedirs(os.path.join("artifacts"), exist_ok=True)
        path = os.path.join("artifacts", "selected_claims_debug.txt")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(f"timestamp: {pd.Timestamp.now()}\n")
            fh.write(f"count: {len(ids)}\n")
            for i, v in enumerate(ids[:10000]):
                fh.write(f"{i+1}\t{v}\n")
    except Exception:
        # best-effort logging; don't fail the app on write errors
        pass


# Clear tracker file once per session start to avoid stale/ghost entries from prior runs.
# This is intentional: the tracker will be reset on first run of the app in this session.
if not st.session_state.get("tracker_cleared_on_start", False):
    try:
        empty = pd.DataFrame(columns=[
            "claim_id","provider_id","flag_types","est_savings",
            "sent","sent_date","channel","letter_path",
            "recouped","recoup_amount","recoup_date","notes"
        ])
        save_tracker(empty)
    except Exception:
        # best-effort; don't crash the app if filesystem is not writable
        pass
    st.session_state["tracker_cleared_on_start"] = True


def home_page():
    st.header("OpenTPA — Payment Integrity Demo")
    st.markdown(
        """
    This demo shows a simple payment integrity workflow: run deterministic rules to flag potential overpayments,
    triage flagged claims, and assemble a set of selected claims for recovery.
        Use the Upload page to stage a CSV (or pick the demo file) before visiting Overview or Filter & Triage.
        """
    )
    st.markdown("---")


def upload_page():
    st.header("Upload Claims")
    raw_dir = os.path.join(os.getcwd(), "data", "raw")
    files = []
    if os.path.isdir(raw_dir):
        files = [f for f in os.listdir(raw_dir) if f.lower().endswith('.csv')]
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Choose demo file from data/raw")
        sel = st.selectbox("Select a file", options=["(use demo)"] + files, index=0)
        if st.button("Load selected file"):
            if sel == "(use demo)":
                df = load_claims_from_raw()
            else:
                df = load_claims_from_raw(sel)
            st.session_state["claims_raw_df"] = df
            st.session_state["active_file"] = sel
            # clear any previously selected claims to avoid ghost selections
            st.session_state["selected_claims"] = []
            # write debug snapshot of selection state
            _write_selected_debug_file([])
            st.session_state.pop("tracker_view_id", None)
            st.session_state.pop("pending_batch", None)
            st.success(f"Loaded {len(df):,} rows from {sel}")

    with col2:
        st.subheader("Or upload your own CSV")
        up = st.file_uploader("Upload claims CSV", type=["csv"], key="upload_home")
        if up is not None:
            df = pd.read_csv(up)
            st.session_state["claims_raw_df"] = df
            # clear selection state when uploading a new file
            st.session_state["selected_claims"] = []
            # write debug snapshot of selection state
            _write_selected_debug_file([])
            st.session_state.pop("tracker_view_id", None)
            st.session_state.pop("pending_batch", None)
            st.session_state["active_file"] = getattr(up, "name", "uploaded")
            st.success(f"Uploaded and staged {len(df):,} rows")


def overview_page():
    st.header("Current Status")
    df_raw = st.session_state.get("claims_raw_df")
    if df_raw is None or df_raw.empty:
        st.info("No claims staged. Go to Home and load or upload a file.")
        return

    df_feat, flags, sav, source_label = get_pipeline_results()
    # prune any stale session selections that don't exist in the processed features
    _prune_session_selected_to_features(df_feat)
    # aggregate flags to one row per claim to avoid duplicating feature rows on merge
    flags_agg = aggregate_flags(flags)
    # merge to create view (use aggregated flags)
    view = df_feat.merge(flags_agg[["claim_id", "flag_types", "reason"]], on="claim_id", how="left")
    view = view.merge(sav, on="claim_id", how="left")
    view["est_savings"] = view.get("est_savings", 0.0).fillna(0.0)

    # ensure reason_group exists for filters (map granular flag types to top-level groups)
    if "flag_types" in view.columns:
        # use the first flag listed (if multiple) to map to a top-level reason group
        view["reason_group"] = view["flag_types"].fillna("").apply(lambda v: reason_group_from_flag(v.split(",")[0]) if v else "Other")
    else:
        view["reason_group"] = "Other"

    # Use the processed features (one row per input claim) for canonical totals
    total_claims = int(df_feat["claim_id"].nunique())
    total_allowed = float(df_feat.get("allowed_amount", 0).fillna(0).sum())
    total_paid = float(df_feat.get("paid_amount", 0).fillna(0).sum())

    # flags is per-claim-per-rule; number of flagged claims should count unique claim_ids
    flagged = flags[flags.get("claim_id").notna()].copy()
    flagged_count = int(flagged["claim_id"].nunique())

    # savings are computed per-claim (rule_savings returns per-claim max); use sav for recoverable totals
    potential_recoverable = float(sav["est_savings"].sum()) if "est_savings" in sav.columns else 0.0
    new_total_paid = total_paid - potential_recoverable

    # Show staged (raw) row count for diagnosis (raw may differ from processed view)
    raw_count = len(df_raw)
    raw_unique = int(df_raw["claim_id"].nunique()) if "claim_id" in df_raw.columns else raw_count

    # First row: current totals
    c1, c2, c3 = st.columns(3)
    c1.metric("Total claims (processed)", f"{total_claims:,}")
    c2.metric("Total allowed $", f"${int(math.floor(total_allowed)):,}")
    c3.metric("Total paid $", f"${int(math.floor(total_paid)):,}")
    st.caption(f"Staged raw rows: {raw_count:,} — unique claim_ids: {raw_unique:,}")

    st.subheader("Recoverable")
    r1, r2, r3 = st.columns(3)
    r1.metric("Flagged claims", f"{flagged_count:,}")
    r2.metric("Potential recoverable $", f"${int(math.floor(potential_recoverable)):,}")
    r3.metric("Paid after recovered $", f"${int(math.floor(new_total_paid)):,}")

    st.markdown("---")
    st.subheader("Flag summary by reason group")
    if flagged.empty:
        st.write("No flagged claims found.")
        return
    # build a richer aggregation for reason groups
    # Build reason-group aggregation on a per-claim basis (deduplicate multiple flags per claim)
    if not flagged.empty:
        # prefer the aggregated flags table for per-claim work
        flags_first = flags_agg.copy()
        flags_first["reason_group"] = flags_first["flag_types"].fillna("").apply(lambda v: reason_group_from_flag(v.split(",")[0]) if v else "Other")
        # join to features and savings to get allowed/paid/est_savings per claim
        reason_base = (
            flags_first[ ["claim_id","reason_group"] ]
            .merge(df_feat[["claim_id","allowed_amount","paid_amount","cpt_code"]], on="claim_id", how="left")
            .merge(sav[["claim_id","est_savings"]], on="claim_id", how="left")
        )
        agg = (
            reason_base.groupby("reason_group", dropna=False)
            .agg(
                total_claims=("claim_id", "nunique"),
                total_allowed=("allowed_amount", "sum"),
                total_paid=("paid_amount", "sum"),
                claims_recoverable=("est_savings", lambda s: int((s.fillna(0) > 0).sum())),
                recoverable_amount=("est_savings", "sum"),
            )
            .reset_index()
            .sort_values("recoverable_amount", ascending=False)
        )

        # show with AgGrid so columns are sortable and index column is hidden
        gb = GridOptionsBuilder.from_dataframe(agg.reset_index(drop=True))
        for c in agg.columns:
            gb.configure_column(c, sortable=True, filter=True)
        grid_opts = gb.build()
        AgGrid(agg.reset_index(drop=True), gridOptions=grid_opts, fit_columns_on_grid_load=True, enable_enterprise_modules=False)
    else:
        st.write("No flagged claims found.")

    st.markdown("---")
    st.subheader("Flag summary by CPT code")
    # CPT table: number of claims and total recoverable per CPT
    # CPT-level aggregation, also computed per-claim (deduplicate claims with multiple flags)
    if not flagged.empty:
        cpt_base = (
            flags_first[["claim_id"]]
            .merge(df_feat[["claim_id","cpt_code","allowed_amount","paid_amount"]], on="claim_id", how="left")
            .merge(sav[["claim_id","est_savings"]], on="claim_id", how="left")
        )
        cpt_agg = (
            cpt_base.assign(cpt_code=lambda d: d.get("cpt_code").astype(str))
            .groupby("cpt_code", dropna=False)
            .agg(
                total_claims=("claim_id", "nunique"),
                total_allowed=("allowed_amount", "sum"),
                total_paid=("paid_amount", "sum"),
                claims_recoverable=("est_savings", lambda s: int((s.fillna(0) > 0).sum())),
                recoverable_amount=("est_savings", "sum"),
            )
            .reset_index()
            .sort_values("recoverable_amount", ascending=False)
        )

        gb2 = GridOptionsBuilder.from_dataframe(cpt_agg.reset_index(drop=True))
        for c in cpt_agg.columns:
            gb2.configure_column(c, sortable=True, filter=True)
        grid_opts2 = gb2.build()
        AgGrid(cpt_agg.reset_index(drop=True), gridOptions=grid_opts2, fit_columns_on_grid_load=True, enable_enterprise_modules=False)
    else:
        st.write("No CPT aggregates to show.")

    # Provider aggregates (one row per claim then grouped by provider)
    if not flagged.empty:
        flags_first = flags_agg.copy()
        prov_base = (
            flags_first[["claim_id"]]
            .merge(df_feat[["claim_id", "provider_id", "allowed_amount", "paid_amount"]], on="claim_id", how="left")
            .merge(sav[["claim_id", "est_savings"]], on="claim_id", how="left")
        )
        prov_agg = (
            prov_base.groupby("provider_id", dropna=False)
            .agg(
                total_allowed=("allowed_amount", "sum"),
                total_paid=("paid_amount", "sum"),
                total_claims=("claim_id", "nunique"),
                claims_recoverable=("est_savings", lambda s: int((s.fillna(0) > 0).sum())),
                recoverable_amount=("est_savings", "sum"),
            )
            .reset_index()
            .sort_values("recoverable_amount", ascending=False)
        )
        st.markdown('---')
        st.subheader("Provider aggregates")
        gbp = GridOptionsBuilder.from_dataframe(prov_agg.reset_index(drop=True))
        for c in prov_agg.columns:
            gbp.configure_column(c, sortable=True, filter=True)
        AgGrid(prov_agg.reset_index(drop=True), gridOptions=gbp.build(), fit_columns_on_grid_load=True, enable_enterprise_modules=False, height=300)


def filter_triage_page():
    st.header("Filter & Triage")
    df_raw = st.session_state.get("claims_raw_df")
    if df_raw is None or df_raw.empty:
        st.info("No claims staged. Go to Home and load or upload a file.")
        return

    df_feat, flags, sav, source_label = get_pipeline_results()
    # aggregate flags into one row per claim for display (prevents duplicated feature rows)
    flags_agg = aggregate_flags(flags)
    view = df_feat.merge(flags_agg[["claim_id", "flag_types", "reason"]], on="claim_id", how="left")
    view = view.merge(sav, on="claim_id", how="left")
    view["est_savings"] = view.get("est_savings", 0.0).fillna(0.0)

    # ensure reason_group exists
    if "flag_types" in view.columns:
        view["reason_group"] = view["flag_types"].fillna("").apply(lambda v: reason_group_from_flag(v.split(",")[0]) if v else "Other")
    else:
        view["reason_group"] = "Other"

    flagged = view[view.get("flag_types").notna()].copy()
    if flagged.empty:
        st.write("No flagged claims to triage.")
        return

    # Instructions span full width, then two columns: left=filters, right=actions+table
    st.markdown("Please select the claims you'd like to attempt recovery on.")
    st.markdown("---")

    # deduplicate flagged claims per-claim for provider and aggregations
    reasons = sorted(flagged["reason_group"].dropna().unique().tolist())
    # compute choices for CPT and provider filters (widgets live in the left column)
    cpt_choices = sorted(flagged.get("cpt_code", pd.Series(dtype=str)).dropna().unique().tolist())
    prov_choices = sorted(flagged.get("provider_id", pd.Series(dtype=str)).dropna().unique().tolist())

    # two-column layout for filters (left) and actions+table (right)
    left, right = st.columns([3, 9])

    # left column: filters only
    with left:
        st.subheader("Filters")
        st.caption("Narrow the flagged claims on the right using these filters")
        ALL_REASONS = "(All reasons)"
        reason_options = [ALL_REASONS] + reasons
        sel_reasons = st.multiselect("Reason groups", reason_options, default=reasons, key="triage_reasons")
        # interpret special '(All reasons)' token: empty or selecting the token means no filtering (all)
        if not sel_reasons or ALL_REASONS in sel_reasons:
            effective_reasons = reasons
        else:
            # remove possible special token if accidentally present
            effective_reasons = [r for r in sel_reasons if r != ALL_REASONS]

        ALL_CPTS = "(All CPTs)"
        cpt_options = [ALL_CPTS] + cpt_choices
        sel_cpts = st.multiselect("CPT codes", cpt_options, default=cpt_choices, key="triage_cpts")
        if not sel_cpts or ALL_CPTS in sel_cpts:
            effective_cpts = cpt_choices
        else:
            effective_cpts = [c for c in sel_cpts if c != ALL_CPTS]

        ALL_PROVS = "(All Providers)"
        prov_options = [ALL_PROVS] + prov_choices
        sel_provs = st.multiselect("Providers", prov_options, default=prov_choices, key="triage_provs")
        if not sel_provs or ALL_PROVS in sel_provs:
            effective_provs = prov_choices
        else:
            effective_provs = [p for p in sel_provs if p != ALL_PROVS]

        min_amt = st.number_input("Min est. recoverable $", min_value=0.0, value=0.0, key="triage_min_amt")
        paid_min, paid_max = st.slider("Paid amount range", float(flagged["paid_amount"].min() if "paid_amount" in flagged.columns else 0.0), float(flagged["paid_amount"].max() if "paid_amount" in flagged.columns else 0.0), (0.0, float(flagged["paid_amount"].max() if "paid_amount" in flagged.columns else 0.0)), key="triage_paid_range")

    # right column: action rows then the filtered table
    with right:
        # First row: Review + Download actions
        d0, d1, d2 = st.columns([1, 1, 1])
        # prepare selected dataframe for download payload
        sel_ids_current = st.session_state.get("selected_claims", [])
        sel_df_payload = view[view["claim_id"].astype(str).isin(sel_ids_current)].copy() if sel_ids_current else pd.DataFrame(columns=view.columns)
        if d0.button("Proceed with Selected Claims"):
            # navigate by setting a query param so the main() can pick it up on the next rerun
            try:
                st.experimental_set_query_params(page="Export Report")
            except Exception:
                # fallback to the goto_page flag if query params are unavailable
                st.session_state["goto_page"] = "Export Report"
            return
        d1.download_button("Download all flagged", flagged.to_csv(index=False).encode("utf-8"), file_name="flagged_all.csv")
        d2.download_button("Download selected", sel_df_payload.to_csv(index=False).encode("utf-8"), file_name="flagged_selected.csv")

        # Second row: selection management
        s1, s2, s3 = st.columns(3)
        # SELECT/DESELECT actions: set a lightweight 'triage_action' flag so the grid JS can run fast client-side APIs
        # Note: selection management buttons removed per request (no select/deselect/clear here)

        # build filtered dataframe for display (use effective lists)
        f = flagged.copy()
        f = f[f["reason_group"].isin(effective_reasons)]
        if "cpt_code" in f.columns:
            f = f[f["cpt_code"].isin(effective_cpts)]
        if "provider_id" in f.columns:
            f = f[f["provider_id"].isin(effective_provs)]
        f = f[f["est_savings"] >= float(min_amt or 0.0)]
        if "paid_amount" in f.columns:
            f = f[(f["paid_amount"] >= paid_min) & (f["paid_amount"] <= paid_max)]

        st.markdown(f"Showing {len(f):,} flagged claims")

        cols = [c for c in ["claim_id", "provider_id", "member_id", "service_date", "cpt_code", "paid_amount", "allowed_amount", "est_savings", "flag_types", "reason"] if c in f.columns]
        display = f[cols].round(2)

        display_for_grid = display.reset_index(drop=True)
        # compute pre-selected row indices based on session selected_claims so checkboxes visually reflect selection
        current_sel = st.session_state.get("selected_claims", [])
        pre_selected_indices = [i for i, r in display_for_grid.iterrows() if str(r.get("claim_id")) in current_sel]

        gb = GridOptionsBuilder.from_dataframe(display_for_grid)
        gb.configure_selection(selection_mode="multiple", use_checkbox=True, pre_selected_rows=pre_selected_indices, header_checkbox=True)
        # make columns sortable and filterable
        for c in display.columns:
            gb.configure_column(c, sortable=True, filter=True)

        # inject a small JS hook that will run fast ag-Grid APIs onGridReady
        # action: 'select_filtered' -> api.selectAllFiltered();
        # action: 'deselect_filtered' -> api.deselectAll();
        sess_sel = st.session_state.get("selected_claims", [])
        action = st.session_state.get("triage_action", None)
        # keep only those selected ids that are visible in the current view
        view_claims = [str(x) for x in display_for_grid.get("claim_id", pd.Series(dtype=str)).tolist()]
        to_select = [s for s in sess_sel if s in view_claims]
        js_array = json.dumps(to_select)
        action_json = json.dumps(action)
        js = JsCode(f"""
        function(params) {{
            const action = {action_json};
            try {{
                if(action === 'select_filtered') {{
                    params.api.selectAllFiltered();
                    return;
                }}
                if(action === 'deselect_filtered') {{
                    params.api.deselectAll();
                    return;
                }}
            }} catch(e) {{}}
            const sel = {js_array};
            if(!sel || sel.length === 0) return;
            params.api.forEachNode(function(node) {{
                try {{
                    if(sel.indexOf(String(node.data.claim_id)) !== -1) {{
                        node.setSelected(true);
                    }}
                }} catch(e) {{}}
            }});
        }}
        """)
        gb.configure_grid_options(onGridReady=js)

        grid_opts = gb.build()
        # calculate a reasonable table height to match filter column
        table_height = min(800, max(300, 30 + len(display) * 24))
        resp = AgGrid(
            display_for_grid,
            gridOptions=grid_opts,
            data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
            update_mode=GridUpdateMode.NO_UPDATE,
            fit_columns_on_grid_load=False,
            height=table_height,
            allow_unsafe_jscode=True,
        )

        # normalize response: selected_rows may be list(dict) or DataFrame or None
        selected_rows = resp.get("selected_rows", None)
        if selected_rows is None:
            selected_rows = []
        elif isinstance(selected_rows, pd.DataFrame):
            selected_rows = selected_rows.to_dict("records")

        # merge with session_state selected_claims (persist across filters)
        sel_ids = [str(r.get("claim_id")).strip() for r in selected_rows if isinstance(r, dict) and r.get("claim_id") is not None]
        if sel_ids:
            # add to existing while preserving order; normalize existing values
            existing = [str(x).strip() for x in st.session_state.get("selected_claims", [])]
            combined = list(dict.fromkeys(existing + sel_ids))
            st.session_state["selected_claims"] = combined
            # persist debug snapshot of the selection for offline diagnostics
            try:
                _write_selected_debug_file(combined)
            except Exception:
                pass
            # prune any selections that do not exist in the processed features (keep list canonical)
            try:
                _prune_session_selected_to_features(df_feat)
            except Exception:
                pass

        # clear any one-shot triage action so it won't re-run on the next unrelated rerun
        if st.session_state.get("triage_action", None) is not None:
            st.session_state["triage_action"] = None

        # Light-weight summary for selected claims to avoid rendering a large table here
        st.markdown("---")
        current_sel = st.session_state.get("selected_claims", [])
        if not current_sel:
            st.write("No claims selected.")
        else:
            sel_df = view[view["claim_id"].astype(str).isin(current_sel)].copy()
            m1, m2 = st.columns(2)
            m1.metric("Selected flagged claims", f"{len(sel_df):,}")
            m2.metric("Selected potential recoverable $", f"${sel_df.get('est_savings', pd.Series(dtype=float)).sum():,.2f}")


def selected_claims_page():
    st.header("Export Report")
    df_raw = st.session_state.get("claims_raw_df")
    sel = st.session_state.get("selected_claims", [])
    if df_raw is None or df_raw.empty:
        st.info("No claims staged. Go to Home and load or upload a file.")
        return
    if not sel:
        st.info("No claims selected. Use Filter & Triage to select flagged claims.")
        return
    # fetch pipeline results (cached)
    df_feat, flags, sav, _ = get_pipeline_results()
    _prune_session_selected_to_features(df_feat)
    # aggregate flags to one row per claim to keep multi-flag metadata and avoid duplication
    flags_agg = aggregate_flags(flags)
    view = df_feat.merge(flags_agg[["claim_id", "flag_types", "reason"]], on="claim_id", how="left")
    view = view.merge(sav, on="claim_id", how="left")
    view["est_savings"] = view.get("est_savings", 0.0).fillna(0.0)

    # Compute ML anomaly flags (IsolationForest) and merge into the view. Cache by pipeline key to avoid repeated fits.
    try:
        cache_key = st.session_state.get("pipeline_cache_key", "")
        ml_cache = st.session_state.get("ml_cache", {}) or {}
        if ml_cache.get("key") != cache_key:
            try:
                ml_res = compute_ml_anomalies(df_feat, pct=0.02)
            except ImportError:
                ml_res = pd.DataFrame()
                try:
                    st.caption("ML anomaly detection disabled: install scikit-learn to enable IsolationForest: pip install scikit-learn")
                except Exception:
                    pass
            st.session_state["ml_cache"] = {"key": cache_key, "result": ml_res}
        else:
            ml_res = ml_cache.get("result", pd.DataFrame())
        if ml_res is not None and not ml_res.empty:
            view = view.merge(ml_res, on="claim_id", how="left")
        else:
            view["ml_score"] = float("nan")
            view["ml_recoup_candidate"] = False
    except Exception:
        view["ml_score"] = float("nan")
        view["ml_recoup_candidate"] = False

    # ensure reason_group exists for selection summaries
    if "flag_types" in view.columns:
        view["reason_group"] = view["flag_types"].fillna("").apply(lambda v: reason_group_from_flag(v.split(",")[0]) if v else "Other")
    else:
        view["reason_group"] = "Other"

    # normalize and deduplicate session selected ids before filtering
    normalized_sel = list(dict.fromkeys([str(x).strip() for x in sel if x is not None]))
    st.session_state["selected_claims"] = normalized_sel
    # persist debug snapshot of the selection for offline diagnostics
    try:
        _write_selected_debug_file(normalized_sel)
    except Exception:
        pass
    sel_df = view[view["claim_id"].astype(str).isin(normalized_sel)].copy()
    # Compute duplicate mappings: for claims that share the duplication key (member_id, provider_id, service_date, cpt_code)
    # create a column "Duplicate Claim Number" which lists the other claim_id(s) in the same duplicate group.
    try:
        dup_key = ["member_id", "provider_id", "service_date", "cpt_code"]
        if all(k in df_feat.columns for k in dup_key + ["claim_id"]):
            # find groups with >1 member
            grp = df_feat.loc[:, ["claim_id"] + dup_key].astype(str).fillna("")
            grp["_dup_group"] = grp[dup_key].agg("||".join, axis=1)
            counts = grp.groupby("_dup_group").agg(n=("claim_id","size")).reset_index()
            multi_groups = set(counts.loc[counts["n"]>1, "_dup_group"].tolist())
            if multi_groups:
                # mapping from claim_id -> other claim ids in same group
                mapping = {}
                for g in multi_groups:
                    members = grp.loc[grp["_dup_group"]==g, "claim_id"].astype(str).tolist()
                    for cid in members:
                        others = [m for m in members if str(m) != str(cid)]
                        mapping[str(cid)] = ",".join(others) if others else ""
                # attach to sel_df
                sel_df["Duplicate Claim Number"] = sel_df["claim_id"].astype(str).map(lambda x: mapping.get(str(x), ""))
            else:
                sel_df["Duplicate Claim Number"] = ""
        else:
            sel_df["Duplicate Claim Number"] = ""
    except Exception:
        # best-effort: if anything fails, create empty column
        sel_df["Duplicate Claim Number"] = ""
    st.metric("Selected flagged claims", f"{len(sel_df):,}")
    st.metric("Selected potential recoverable $", f"${sel_df['est_savings'].sum():,.2f}")
    # ML recoupment candidates in current selection
    try:
        ml_count = int(sel_df.get("ml_recoup_candidate", pd.Series(dtype=bool)).sum()) if not sel_df.empty else 0
        st.metric("ML recoup candidates", f"{ml_count}")
    except Exception:
        pass

    # Top-of-page export: assemble metadata and present a client/attestation form, then export buttons
    rpt_col_left, rpt_col_right = st.columns([1, 1])
    # derive analysis period from service_date if available
    analysis_period = ""
    try:
        if "service_date" in df_feat.columns:
            dates = pd.to_datetime(df_feat["service_date"], errors="coerce")
            if not dates.dropna().empty:
                analysis_period = f"{dates.min().date().isoformat()} to {dates.max().date().isoformat()}"
    except Exception:
        analysis_period = ""

    data_file = st.session_state.get("active_file", "") or ""
    data_file_mtime = None
    try:
        if data_file:
            path = os.path.join(os.getcwd(), "data", "raw", data_file)
            if os.path.exists(path):
                data_file_mtime = datetime.utcfromtimestamp(os.path.getmtime(path)).isoformat() + "Z"
    except Exception:
        data_file_mtime = None

    cache_key = st.session_state.get("pipeline_cache_key", "")
    meta = {
        "title": "Fiduciary Report - Selected Claims",
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "n_staged": 0 if df_raw is None else len(df_raw),
        "n_selected": len(sel_df),
        "total_recoverable": float(sel_df.get('est_savings', pd.Series(dtype=float)).sum()),
        "data_file": data_file,
        "data_file_mtime": data_file_mtime,
        "analysis_period": analysis_period,
        "pipeline_key": cache_key,
        "source_label": _ if _ is not None else "",
        "methodology": "Rules-based flagging and recoverable estimation using configured rule set. See Rules Applied section for counts.",
    }

    # prepare rules summary (counts per flag_type)
    try:
        rule_agg = flags.groupby('flag_type').size().reset_index(name='count').sort_values('count', ascending=False)
    except Exception:
        rule_agg = pd.DataFrame()
    # prepare full flags CSV for optional appendix
    try:
        flags_csv = flags.to_csv(index=False).encode("utf-8") if (flags is not None and not flags.empty) else b"claim_id\n"
    except Exception:
        flags_csv = b"claim_id\n"
    # prepare small aggregates for inclusion in the report
    reason_agg = sel_df.groupby("reason_group").agg(n_claims=("claim_id","nunique"), recoverable=("est_savings","sum")).reset_index().sort_values("recoverable", ascending=False)
    prov_base = (
        sel_df[["claim_id"]]
        .merge(df_feat[["claim_id", "provider_id", "allowed_amount", "paid_amount"]], on="claim_id", how="left")
        .merge(sav[["claim_id", "est_savings"]], on="claim_id", how="left")
    )
    prov_agg = (
        prov_base.groupby("provider_id", dropna=False)
        .agg(
            total_allowed=("allowed_amount", "sum"),
            total_paid=("paid_amount", "sum"),
            total_claims=("claim_id", "nunique"),
            claims_recoverable=("est_savings", lambda s: int((s.fillna(0) > 0).sum())),
            recoverable_amount=("est_savings", "sum"),
        )
        .reset_index()
        .sort_values("recoverable_amount", ascending=False)
    )

    # ML-only candidates (anomaly-detected but not flagged by rules)
    try:
        ml_only_df = pd.DataFrame()
        if "ml_recoup_candidate" in view.columns:
            ml_only_df = view[ (view.get("ml_recoup_candidate") == True) & (view.get("flag_types").fillna("") == "") ].copy()
        else:
            ml_only_df = pd.DataFrame()
    except Exception:
        ml_only_df = pd.DataFrame()

    # No interactive report metadata on the Export Report page (keep exports minimal)
    meta["client_name"] = ""
    meta["client_id"] = ""
    meta["contact_name"] = ""
    meta["contact_email"] = ""
    meta["signer_name"] = ""
    meta["signer_title"] = ""
    meta["signature_date"] = ""

    # no logo support (not collecting metadata here)
    logo_bytes = None

    # Prepare sensible filenames (generic client) containing a timestamp
    safe_client = "client"
    stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    # Offer a single Excel download that contains three sheets (Selected Claims, Providers, Reason Groups)
    excel_name = f"fiduciary_report_{safe_client}_{stamp}.xlsx"
    try:
        excel_bytes = _build_fiduciary_excel(sel_df, prov_agg, reason_agg, ml_df=ml_only_df)
        clicked_excel = st.download_button(
            "Download Excel (Selected Claims, Providers, Reason Groups)",
            data=excel_bytes,
            file_name=excel_name,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        if clicked_excel:
            # ensure we remain on Export Report after the download-triggered rerun
            st.session_state["goto_page"] = "Export Report"
    except ImportError:
        st.caption("To enable Excel export install openpyxl: pip install openpyxl")
    except Exception as e:
        st.error(f"Failed to generate Excel: {e}")
    docx_name = f"fiduciary_report_{safe_client}_{stamp}.docx"
    pdf_name = f"fiduciary_report_{safe_client}_{stamp}.pdf"

    with rpt_col_left:
        try:
            data = _build_fiduciary_docx(sel_df, prov_agg, reason_agg, meta, rule_agg=rule_agg, logo_bytes=logo_bytes)
            clicked_docx = st.download_button("Download report (.docx)", data=data, file_name=docx_name, mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
            if clicked_docx:
                st.session_state["goto_page"] = "Export Report"
        except ImportError:
            st.caption("Download report (.docx): install python-docx to enable this export: pip install python-docx")
        except Exception as e:
            st.error(f"Failed to generate .docx: {e}")
    with rpt_col_right:
        try:
            data = _build_fiduciary_pdf(sel_df, prov_agg, reason_agg, meta, rule_agg=rule_agg, logo_bytes=logo_bytes)
            clicked_pdf = st.download_button("Download report (.pdf)", data=data, file_name=pdf_name, mime="application/pdf")
            if clicked_pdf:
                st.session_state["goto_page"] = "Export Report"
        except ImportError:
            st.caption("Download report (.pdf): install reportlab to enable this export: pip install reportlab")
        except Exception as e:
            st.error(f"Failed to generate PDF: {e}")

    st.markdown("### Top reason groups in selection")
    reason_agg = sel_df.groupby("reason_group").agg(n_claims=("claim_id","nunique"), recoverable=("est_savings","sum")).reset_index().sort_values("recoverable", ascending=False)
    # format recoverable column as whole-dollar strings
    reason_disp = _format_money_columns(reason_agg.copy(), ['recoverable'])
    st.dataframe(reason_disp, height=200)

    st.markdown("### Selected claims")
    cols = [c for c in ["claim_id","provider_id","member_id","service_date","cpt_code","paid_amount","allowed_amount","est_savings","reason"] if c in sel_df.columns]
    sel_disp = sel_df[cols].copy()
    sel_disp = _format_money_columns(sel_disp, ['allowed_amount', 'paid_amount', 'est_savings'])
    st.dataframe(sel_disp, height=300)

    # Show ML-only anomaly candidates (not flagged by rules)
    try:
        if not ml_only_df.empty:
            st.markdown("### ML-only anomaly candidates (not flagged by rules)")
            ml_cols = [c for c in ["claim_id","provider_id","member_id","service_date","cpt_code","paid_amount","allowed_amount","est_savings","ml_score"] if c in ml_only_df.columns]
            ml_disp = ml_only_df[ml_cols].copy()
            ml_disp = _format_money_columns(ml_disp, ['allowed_amount', 'paid_amount', 'est_savings'])
            st.dataframe(ml_disp, height=300)
        else:
            st.markdown("### ML-only anomaly candidates (not flagged by rules)")
            st.write("No ML-only anomaly candidates were identified.")
    except Exception:
        pass

    st.markdown("### Providers (NPI-like)")
    # Provider aggregates for selected claims
    prov_base = (
        sel_df[["claim_id"]]
        .merge(df_feat[["claim_id", "provider_id", "allowed_amount", "paid_amount"]], on="claim_id", how="left")
        .merge(sav[["claim_id", "est_savings"]], on="claim_id", how="left")
    )
    prov_agg = (
        prov_base.groupby("provider_id", dropna=False)
        .agg(
            total_allowed=("allowed_amount", "sum"),
            total_paid=("paid_amount", "sum"),
            total_claims=("claim_id", "nunique"),
            claims_recoverable=("est_savings", lambda s: int((s.fillna(0) > 0).sum())),
            recoverable_amount=("est_savings", "sum"),
        )
        .reset_index()
        .sort_values("recoverable_amount", ascending=False)
    )
    if not prov_agg.empty:
        prov_disp = _format_money_columns(prov_agg.copy(), ['total_allowed', 'total_paid', 'recoverable_amount'])
        st.dataframe(prov_disp, height=300)
    else:
        st.write("No provider aggregates for selected claims.")

    # Navigation: provide an explicit button to go to the Recovery Tracker
    # Recovery tracker removed (simplify UI); use Export Report for downstream work


def recovery_tracker_page():
    st.header("Recovery Tracker")
    tracker = load_tracker()
    # selection manager keys for master-detail mode
    st.session_state.setdefault("tracker_selected_ids", [])
    st.session_state.setdefault("tracker_view_id", None)
    st.session_state.setdefault("tracker_page", 0)
    # union of current session-selected claims and any claims present in the tracker
    sel_session = set([str(x).strip() for x in st.session_state.get("selected_claims", [])])
    tracker_ids = set(tracker["claim_id"].astype(str).str.strip().tolist()) if not tracker.empty else set()
    sel = list(sorted(sel_session.union(tracker_ids)))

    # compute overview metrics
    st.markdown("### Overview")
    total_selected = len(sel)
    # compute recoverable sums and counts by status (for all selected claims)
    sum_no = sum_started = sum_finished = 0.0
    no_progress = started = finished = 0
    if sel:
        try:
            df_raw = st.session_state.get("claims_raw_df")
            df_feat, flags, sav, _ = get_pipeline_results()
            # aggregate flags for one-row-per-claim merges
            flags_agg = aggregate_flags(flags)
            # ensure session selection only contains valid IDs
            _prune_session_selected_to_features(df_feat)
            view_all = df_feat.merge(flags_agg[["claim_id", "flag_types", "reason"]], on="claim_id", how="left")
            view_all = view_all.merge(sav, on="claim_id", how="left")
            view_all["claim_id"] = view_all["claim_id"].astype(str)
            sel_view = view_all[view_all["claim_id"].isin(sel)].copy()
            if not tracker.empty:
                tr_idx = tracker.set_index(tracker["claim_id"].astype(str).str.strip())
                def _status(cid: str):
                    if cid in tr_idx.index:
                        row = tr_idx.loc[cid]
                        if bool(row.get("recouped") == True):
                            return "Recouped"
                        if bool(row.get("sent") == True):
                            return "In progress"
                        return "No Progress"
                    return "No Progress"
                sel_view["tracker_status"] = sel_view["claim_id"].apply(_status)
            else:
                sel_view["tracker_status"] = "No Progress"
            # compute counts and sums from sel_view so session-only claims count as No progress by default
            started = int((sel_view["tracker_status"] == "In progress").sum())
            finished = int((sel_view["tracker_status"] == "Recouped").sum())
            sum_no = float(sel_view.loc[sel_view["tracker_status"] == "No Progress", "est_savings"].sum())
            sum_started = float(sel_view.loc[sel_view["tracker_status"] == "In progress", "est_savings"].sum())
            sum_finished = float(sel_view.loc[sel_view["tracker_status"] == "Recouped", "est_savings"].sum())
        except Exception:
            sum_no = sum_started = sum_finished = 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Selected claims", f"{total_selected}")
    c2.metric("No Progress", f"{no_progress} — ${sum_no:,.2f}")
    c3.metric("In progress", f"{started} — ${sum_started:,.2f}")
    c4.metric("Recouped", f"{finished} — ${sum_finished:,.2f}")

    # compact toolbar: batch actions and export
    toolbar_left, toolbar_mid, toolbar_mid2, toolbar_right = st.columns([3, 1, 1, 1])
    with toolbar_left:
        st.caption("Batch actions on selected rows (select with checkboxes)")
    with toolbar_mid:
        if st.button("Batch Mark No Progress"):
            st.session_state["batch_requested"] = {"action": "no_progress"}
    with toolbar_mid2:
        if st.button("Batch Add to tracker"):
            st.session_state["batch_requested"] = {"action": "add"}
    with toolbar_right:
        if st.button("Batch Mark In progress"):
            st.session_state["batch_requested"] = {"action": "sent"}
    # Admin: Reset entire tracker statuses to No progress (confirmation required)
    st.markdown("---")
    reset_col, _ = st.columns([1, 3])
    with reset_col:
        if st.button("Reset all tracker statuses to No Progress"):
            st.session_state["pending_reset_all"] = True
    if st.session_state.get("pending_reset_all"):
        st.warning("This will mark every row in the tracker as No Progress (clears sent/recouped flags). Confirm?")
        r_yes, r_no = st.columns([1, 1])
        with r_yes:
            if st.button("Confirm reset all"):
                tr = load_tracker()
                for cid in tr["claim_id"].astype(str).tolist():
                    try:
                        upsert_tracker({"claim_id": cid, "sent": False, "sent_date": pd.NA, "recouped": False, "recoup_date": pd.NA, "recoup_amount": 0.0})
                    except Exception:
                        pass
                st.success("Reset tracker statuses to No Progress")
                st.session_state.pop("pending_reset_all", None)
        with r_no:
            if st.button("Cancel"):
                st.session_state.pop("pending_reset_all", None)

    # Top-header export: export all selected claims with tracker metadata
    export_csv = "claim_id\n"  # default empty CSV
    if sel:
        try:
            df_raw = st.session_state.get("claims_raw_df")
            df_feat, flags, sav, _ = get_pipeline_results()
            cache = st.session_state.get("pipeline_cache", {}) or {}
            flags_agg = cache.get("flags_agg")
            if flags_agg is None or (isinstance(flags_agg, pd.DataFrame) and flags_agg.empty):
                flags_agg = aggregate_flags(flags)
            view_all = df_feat.merge(flags_agg[["claim_id", "flag_types", "reason"]], on="claim_id", how="left")
            view_all = view_all.merge(sav, on="claim_id", how="left")
            view_all["claim_id"] = view_all["claim_id"].astype(str)
            export_df = view_all[view_all["claim_id"].isin(sel)].copy()
            if not tracker.empty:
                export_df = export_df.merge(tracker, left_on="claim_id", right_on=tracker["claim_id"].astype(str), how="left", suffixes=("","_trk"))
            export_csv = export_df.to_csv(index=False)
        except Exception:
            export_csv = "claim_id\n"
    st.download_button("Export claims + recovery status", data=export_csv, file_name="selected_claims_tracker.csv", mime="text/csv")

    st.markdown("---")
    left, right = st.columns([2, 3])

    # left: list of selected claims with status filter and export
    with left:
        st.subheader("Selected flagged claims")
        # master-detail mode toggle
        simple_mode = st.checkbox("Use master-detail list (recommended)", value=True, key="tracker_simple_mode")
        # convenience selection helpers
        sel_helpers_col1, sel_helpers_col2, sel_helpers_col3 = st.columns([1, 1, 1])
        with sel_helpers_col1:
            if st.button("Select visible page"):
                # select visible page (when paginating below)
                # will be handled in the master-detail rendering
                st.session_state["tracker_select_visible"] = True
        with sel_helpers_col2:
            if st.button("Select filtered set"):
                # select all currently filtered rows
                ids = filtered["claim_id"].astype(str).tolist() if 'filtered' in locals() else []
                st.session_state["tracker_selected_ids"] = ids
                _write_selected_debug_file(ids)
        with sel_helpers_col3:
            if st.button("Clear selection"):
                st.session_state["tracker_selected_ids"] = []
                _write_selected_debug_file([])
        # Batch actions available in the toolbar above (select rows using checkboxes first)
        if not sel:
            st.write("No selected claims — use Filter & Triage to add to selection.")
        else:
            df_raw = st.session_state.get("claims_raw_df")
            df_feat, flags, sav, _ = get_pipeline_results()
            cache = st.session_state.get("pipeline_cache", {}) or {}
            flags_agg = cache.get("flags_agg")
            if flags_agg is None or (isinstance(flags_agg, pd.DataFrame) and flags_agg.empty):
                flags_agg = aggregate_flags(flags)
            view = df_feat.merge(flags_agg[["claim_id", "flag_types", "reason"]], on="claim_id", how="left")
            view = view.merge(sav, on="claim_id", how="left")
            view["claim_id"] = view["claim_id"].astype(str)
            sel_df = view[view["claim_id"].isin(sel)].copy()

            # join tracker info (if present) and compute a simple status
            if not tracker.empty:
                tr_idx = tracker.set_index(tracker["claim_id"].astype(str).str.strip())
                def _status(cid: str):
                    if cid in tr_idx.index:
                        row = tr_idx.loc[cid]
                        if bool(row.get("recouped") == True):
                            return "Recouped"
                        if bool(row.get("sent") == True):
                            return "In progress"
                        return "No Progress"
                    return "No progress"
                sel_df["tracker_status"] = sel_df["claim_id"].apply(_status)
            else:
                sel_df["tracker_status"] = "No Progress"

            # status filter
            status_opts = ["No Progress", "In progress", "Recouped"]
            picked = st.multiselect("Status", status_opts, default=status_opts, key="tracker_status_filter")
            filtered = sel_df[sel_df["tracker_status"].isin(picked)].copy()
            # determine visible columns for the grid and create the display dataframe
            base_cols = ["claim_id", "service_date", "provider_id", "cpt_code", "est_savings", "tracker_status"]
            display_cols = [c for c in base_cols if c in filtered.columns]
            display_for_grid = filtered[display_cols].reset_index(drop=True)
            # include a truncated flag_types column for display (full flag_types still available in sel_df)
            if "flag_types" in sel_df.columns:
                def _short(s: str):
                    try:
                        s = str(s)
                    except Exception:
                        s = ""
                    if len(s) > 40:
                        return s[:37] + "..."
                    return s
                # map full flag_types from sel_df into the display frame and truncate for compact display
                fk = {}
                for cid, ft in zip(sel_df["claim_id"].astype(str), sel_df.get("flag_types", pd.Series(dtype=str)).fillna("")):
                    fk[str(cid)] = _short(ft)
                display_for_grid["flag_types_short"] = display_for_grid["claim_id"].astype(str).map(lambda x: fk.get(str(x), ""))
                # try to insert flag_types_short after est_savings if present
                if "flag_types_short" not in display_cols:
                    # place it before tracker_status if tracker_status exists
                    insert_at = display_for_grid.columns.get_loc("tracker_status") if "tracker_status" in display_for_grid.columns else len(display_for_grid.columns)
                    # move the column into the desired position by reordering
                    cols = list(display_for_grid.columns)
                    cols.remove("flag_types_short")
                    cols.insert(insert_at, "flag_types_short")
                    display_for_grid = display_for_grid[cols]
            # add an editable 'view' column (checkbox-like) to control which claim is shown in the detail pane
            display_for_grid["view"] = False
            last_selected = st.session_state.get("tracker_view_id", None)
            if last_selected is not None:
                display_for_grid.loc[display_for_grid["claim_id"].astype(str) == str(last_selected), "view"] = True
            # include editable notes column by joining tracker notes (if present)
            if not tracker.empty:
                notes_map = {str(r): v for r, v in zip(tracker["claim_id"].astype(str), tracker.get("notes", pd.Series(dtype=str)).fillna("").tolist())}
            else:
                notes_map = {}
            display_for_grid["notes"] = display_for_grid["claim_id"].astype(str).map(lambda x: notes_map.get(str(x), ""))

            # If using simple master-detail mode, render a paginated checkbox list instead of the AgGrid
            if simple_mode:
                # pagination
                page_size = 50
                total = len(display_for_grid)
                total_pages = max(1, (total + page_size - 1) // page_size)
                page = st.session_state.get("tracker_page", 0)
                nav_col1, nav_col2, nav_col3 = st.columns([1, 2, 1])
                with nav_col1:
                    if st.button("Prev") and page > 0:
                        st.session_state["tracker_page"] = page - 1
                with nav_col3:
                    if st.button("Next") and page < total_pages - 1:
                        st.session_state["tracker_page"] = page + 1
                st.write(f"Page {page+1} / {total_pages} — showing {min(page_size, max(0, total - page*page_size))} rows")
                start = page * page_size
                end = start + page_size
                page_df = display_for_grid.iloc[start:end]
                # handle "Select visible page" helper (one-shot)
                if st.session_state.pop("tracker_select_visible", False):
                    ids = page_df["claim_id"].astype(str).tolist()
                    st.session_state["tracker_selected_ids"] = list(dict.fromkeys(st.session_state.get("tracker_selected_ids", []) + ids))
                    _write_selected_debug_file(st.session_state["tracker_selected_ids"])
                sel_ids = set(st.session_state.get("tracker_selected_ids", []))
                for _, row in page_df.iterrows():
                    cid = str(row.get("claim_id"))
                    cols = st.columns([0.05, 0.6, 0.2, 0.15])
                    with cols[0]:
                        chk = st.checkbox("", value=(cid in sel_ids), key=f"tracker_chk_{cid}")
                    with cols[1]:
                        st.write(f"**{cid}** — {row.get('provider_id', '')} — ${float(row.get('est_savings',0) or 0):,.2f}")
                        if "flag_types" in row.index and pd.notna(row.get("flag_types")):
                            st.caption(str(row.get("flag_types")))
                    with cols[2]:
                        if st.button("View", key=f"view_btn_{cid}"):
                            st.session_state["tracker_view_id"] = cid
                    with cols[3]:
                        # small status display
                        st.write(row.get("tracker_status", ""))
                    # persist checkbox change
                    if chk and cid not in sel_ids:
                        sel_ids.add(cid)
                        st.session_state["tracker_selected_ids"] = list(sel_ids)
                        _write_selected_debug_file(st.session_state["tracker_selected_ids"])
                    if not chk and cid in sel_ids:
                        sel_ids.discard(cid)
                        st.session_state["tracker_selected_ids"] = list(sel_ids)
                        _write_selected_debug_file(st.session_state["tracker_selected_ids"])
                # if a one-shot batch request was made at the top, materialize it into pending_batch now using the selection manager
                batch_req = st.session_state.pop("batch_requested", None) if st.session_state.get("batch_requested") is not None else None
                if batch_req is not None:
                    batch_ids = st.session_state.get("tracker_selected_ids", [])
                    if batch_ids:
                        st.session_state["pending_batch"] = {"action": batch_req.get("action"), "ids": batch_ids}
                        st.success(f"Queued {batch_req.get('action')} on {len(batch_ids)} claims")
                    else:
                        st.warning("No rows selected — select checkboxes to apply the batch action")
            else:
                gb = GridOptionsBuilder.from_dataframe(display_for_grid)
                # allow multi-row selection for batch actions; pre-select last picked claim across reruns
                last_selected = st.session_state.get("tracker_view_id", None)
                pre_selected_indices = []
                if last_selected is not None:
                    pre_selected_indices = [i for i, r in display_for_grid.reset_index(drop=True).iterrows() if str(r.get("claim_id")) == str(last_selected)]
                # checkbox selection is independent from row click; clicking row won't toggle selection
                gb.configure_grid_options(suppressRowClickSelection=True)
                gb.configure_selection(selection_mode="multiple", use_checkbox=True, pre_selected_rows=pre_selected_indices, header_checkbox=True)
                # make notes editable inline
                if "notes" in display_for_grid.columns:
                    gb.configure_column("notes", editable=True)
                # make the view column editable so users can toggle which claim to view in the right pane
                if "view" in display_for_grid.columns:
                    gb.configure_column("view", editable=True, cellEditor="agCheckboxCellEditor")
                for c in display_for_grid.columns:
                    gb.configure_column(c, sortable=True, filter=True)
                grid_opts = gb.build()
                table_height = min(600, max(200, 30 + len(display_for_grid) * 28))
                resp = AgGrid(display_for_grid, gridOptions=grid_opts, data_return_mode=DataReturnMode.FILTERED_AND_SORTED, height=table_height, allow_unsafe_jscode=True)

                # detect inline-edited notes and view toggles (st_aggrid may return full data or updated_rows)
                edited = resp.get("data", None)
                if edited is None:
                    edited = resp.get("updated_rows", None)
                if edited is not None:
                    # normalize into list of dicts
                    if isinstance(edited, pd.DataFrame):
                        edited = edited.to_dict("records")
                    if isinstance(edited, list):
                        for rec in edited:
                            try:
                                cid = str(rec.get("claim_id"))
                                new_notes = rec.get("notes", "")
                                # only persist if different from tracker
                                old_notes = notes_map.get(cid, "")
                                if new_notes is not None and str(new_notes) != str(old_notes):
                                    upsert_tracker({"claim_id": cid, "notes": new_notes})
                                    # update local map so subsequent rows see the change
                                    notes_map[cid] = new_notes
                                # handle view toggles: if a row's 'view' got toggled True, make it the detail row
                                view_flag = rec.get("view", None)
                                if view_flag:
                                    # set the tracker_view_id to this claim_id
                                    st.session_state["tracker_view_id"] = cid
                                else:
                                    # if user cleared a view and it matches current detail, clear it
                                    if st.session_state.get("tracker_view_id") == cid:
                                        st.session_state.pop("tracker_view_id", None)
                            except Exception:
                                pass

            # selected rows (from checkboxes) - only relevant when using the AgGrid path
            if not simple_mode:
                selected_rows = resp.get("selected_rows", None)
                if selected_rows is None:
                    selected_rows = []
                elif isinstance(selected_rows, pd.DataFrame):
                    selected_rows = selected_rows.to_dict("records")
                # selected_rows is a list of dicts (or empty)
                batch_ids = [str(r.get("claim_id")) for r in selected_rows if isinstance(r, dict) and r.get("claim_id")]
                # if a one-shot batch request was made at the top, materialize it into pending_batch now that we have selected ids
                batch_req = st.session_state.pop("batch_requested", None) if st.session_state.get("batch_requested") is not None else None
                if batch_req is not None:
                    if batch_ids:
                        st.session_state["pending_batch"] = {"action": batch_req.get("action"), "ids": batch_ids}
                        st.success(f"Queued {batch_req.get('action')} on {len(batch_ids)} claims")
                    else:
                        st.warning("No rows selected — select checkboxes to apply the batch action")

            # detail pane selection is driven by the explicit 'view' toggle, stored in session
            selected_claim_id = st.session_state.get("tracker_view_id", None)

            # confirmation for pending batch actions
            pending = st.session_state.get("pending_batch")
            if pending:
                act = pending.get("action")
                ids = pending.get("ids", [])
                st.warning(f"Pending batch action: {act} on {len(ids)} claims")
                c_yes, c_no = st.columns([1, 1])
                with c_yes:
                    if st.button("Confirm batch action"):
                        if act == "add":
                            added = 0
                            for cid in ids:
                                if cid in sel_df["claim_id"].astype(str).values:
                                    r = sel_df[sel_df["claim_id"].astype(str) == cid].iloc[0]
                                    newrow = {
                                        "claim_id": str(r.get("claim_id")),
                                        "provider_id": r.get("provider_id"),
                                        "flag_types": r.get("flag_types", r.get("flag_type", "")),
                                        "est_savings": float(r.get("est_savings", 0) or 0),
                                        "sent": False,
                                        "recouped": False,
                                        "letter_path": "",
                                        "sent_date": pd.NA,
                                        "recoup_date": pd.NA,
                                        "notes": "",
                                    }
                                    upsert_tracker(newrow)
                                    added += 1
                            st.success(f"Added {added} claims to tracker")
                        elif act == "sent":
                            for cid in ids:
                                upsert_tracker({"claim_id": cid, "sent": True, "sent_date": pd.Timestamp.now()})
                            st.success(f"Marked {len(ids)} claims In progress")
                        elif act == "recoup":
                            for cid in ids:
                                rec_amt = 0.0
                                if cid in sel_df["claim_id"].astype(str).values:
                                    rr = sel_df[sel_df["claim_id"].astype(str) == cid].iloc[0]
                                    rec_amt = float(rr.get("est_savings", 0) or 0)
                                upsert_tracker({"claim_id": cid, "recouped": True, "recoup_date": pd.Timestamp.now(), "recoup_amount": rec_amt})
                            st.success(f"Marked {len(ids)} claims Recouped")
                        elif act == "no_progress":
                            for cid in ids:
                                # mark as not started: clear sent/recouped flags and dates
                                upsert_tracker({"claim_id": cid, "sent": False, "sent_date": pd.NA, "recouped": False, "recoup_date": pd.NA, "recoup_amount": 0.0})
                            st.success(f"Marked {len(ids)} claims No Progress")
                        # clear pending
                        st.session_state.pop("pending_batch", None)
                with c_no:
                    if st.button("Cancel batch action"):
                        st.session_state.pop("pending_batch", None)

            # export filtered selected claims with tracker metadata and provider contact
            export_df = filtered.copy()
            contact_cols = ["provider_name", "address1", "city", "state", "zip", "phone", "fax", "email"]
            for col in contact_cols:
                export_df[col] = ""
            for i, r in export_df.iterrows():
                pid = r.get("provider_id")
                try:
                    contact = provider_contact(str(pid)) if pid is not None else {}
                except Exception:
                    contact = {}
                for col in contact_cols:
                    export_df.at[i, col] = contact.get(col, "")
            if not tracker.empty:
                export_df = export_df.merge(tracker, left_on="claim_id", right_on=tracker["claim_id"].astype(str), how="left", suffixes=("","_trk"))
            csv = export_df.to_csv(index=False)
            st.download_button("Export filtered selection (with contact)", data=csv, file_name="selected_claims_tracker_filtered.csv", mime="text/csv")

    # right: detail drill-down for chosen claim and per-claim actions
    with right:
        st.subheader("Claim detail & actions")
        if not sel:
            st.write("No claims selected.")
        else:
            if not tracker.empty:
                tr = tracker.set_index(tracker["claim_id"].astype(str).str.strip())
            else:
                tr = pd.DataFrame()

            if selected_claim_id is None:
                st.write("Pick a claim on the left to see details and take actions.")
            else:
                row = sel_df[sel_df["claim_id"] == selected_claim_id].iloc[0]
                # show key fields
                st.markdown(f"**Claim ID:** {row['claim_id']}")
                st.markdown(f"**Provider:** {row.get('provider_id', '')}")
                st.markdown(f"**Service date:** {row.get('service_date', '')}")
                st.markdown(f"**CPT:** {row.get('cpt_code', '')}")
                try:
                    est = int(math.floor(float(row.get('est_savings', 0) or 0)))
                except Exception:
                    est = row.get('est_savings', 0)
                st.markdown(f"**Estimated recoverable:** ${est:,}")
                st.markdown(f"**Reason:** {row.get('reason', '')}")

                # tracker metadata
                in_tracker = False
                tracker_row = None
                if not tr.empty and selected_claim_id in tr.index:
                    in_tracker = True
                    tracker_row = tr.loc[selected_claim_id]
                    # show a compact tracker status badge and key metadata
                    try:
                        if bool(tracker_row.get("recouped") == True):
                            status_str = "Recouped"
                        elif bool(tracker_row.get("sent") == True):
                            status_str = "In progress"
                        else:
                            status_str = "No Progress"
                    except Exception:
                        status_str = "No Progress"
                    st.markdown("---")
                    if status_str == "Recouped":
                        st.success(f"Status: {status_str}")
                    elif status_str == "In progress":
                        st.info(f"Status: {status_str}")
                    else:
                        st.warning(f"Status: {status_str}")
                    # show compact metadata
                    sent_date = tracker_row.get("sent_date", "")
                    recoup_date = tracker_row.get("recoup_date", "")
                    recoup_amount = tracker_row.get("recoup_amount", 0.0) if hasattr(tracker_row, 'get') else 0.0
                    st.write(f"**Sent:** {sent_date}\n**Recouped:** {recoup_date}\n**Recoup amount:** ${float(recoup_amount or 0):,.2f}")

                # notes editor
                notes_key = f"notes_{selected_claim_id}"
                existing_notes = ""
                if tracker_row is not None and isinstance(tracker_row, (dict,)):
                    existing_notes = tracker_row.get("notes", "")
                elif tracker_row is not None:
                    # if tracker_row is a Series
                    existing_notes = tracker_row.get("notes", "") if "notes" in tracker_row.index else ""
                notes_text = st.text_area("Notes", value=existing_notes, key=notes_key, height=120)
                if st.button("Save Notes"):
                    upsert_tracker({"claim_id": selected_claim_id, "notes": notes_text})
                    st.success("Saved notes")

                # action buttons
                col1, col2, col3 = st.columns(3)
                with col1:
                    if not in_tracker:
                        if st.button("Add to tracker"):
                            newrow = {
                                "claim_id": str(row.get("claim_id")),
                                "provider_id": row.get("provider_id"),
                                "flag_types": row.get("flag_types", row.get("flag_type", "")),
                                "est_savings": float(row.get("est_savings", 0) or 0),
                                "sent": False,
                                "recouped": False,
                                "letter_path": "",
                                "sent_date": pd.NA,
                                "recoup_date": pd.NA,
                                "notes": "",
                            }
                            upsert_tracker(newrow)
                            st.success("Added claim to tracker")
                    else:
                        if st.button("Mark Sent"):
                            upsert_tracker({"claim_id": selected_claim_id, "sent": True, "sent_date": pd.Timestamp.now()})
                            st.success("Marked In progress")
                with col2:
                    if in_tracker:
                        if st.button("Mark Recouped"):
                            upsert_tracker({"claim_id": selected_claim_id, "recouped": True, "recoup_date": pd.Timestamp.now(), "recoup_amount": float(row.get("est_savings", 0) or 0)})
                            st.success("Marked Recouped")
                with col3:
                    # generate letter
                    if st.button("Generate letter"):
                        payer_profile = {
                            "payer_name": "Demo Payer",
                            "payer_address1": "100 Demo Way",
                            "payer_city": "Demo City",
                            "payer_state": "NY",
                            "payer_zip": "10001",
                        }
                        contact = provider_contact(str(row.get("provider_id")))
                        recoup_amount = float(row.get("est_savings", 0) or 0)
                        text = render_letter_text(row.to_dict(), contact, row.get("reason", ""), recoup_amount, payer_profile)
                        path = write_letter_file(text, str(row.get("claim_id")))
                        upsert_tracker({"claim_id": selected_claim_id, "letter_path": path, "recoup_amount": recoup_amount})
                        st.success(f"Wrote letter to {path}")
                        with open(path, "r", encoding="utf-8") as fh:
                            st.download_button("Download letter", data=fh.read(), file_name=os.path.basename(path))

    # Diagnostics: compare sets of claim_ids across triage (flags), session selection, and tracker
    with st.expander("Diagnostics — claim set comparisons", expanded=False):
        df_raw = st.session_state.get("claims_raw_df")
        if df_raw is None or df_raw.empty:
            st.write("No staged claims to analyze.")
        else:
            try:
                df_feat, flags, sav, _ = get_pipeline_results()
                # use aggregated flags (one row per claim) for triage membership checks
                cache = st.session_state.get("pipeline_cache", {}) or {}
                flags_agg = cache.get("flags_agg")
                if flags_agg is None or (isinstance(flags_agg, pd.DataFrame) and flags_agg.empty):
                    flags_agg = aggregate_flags(flags)
                triage_claims = set([])
                if flags_agg is not None and not flags_agg.empty and "claim_id" in flags_agg.columns:
                    triage_claims = set(flags_agg["claim_id"].astype(str).str.strip().unique())
                session_claims = set([str(x).strip() for x in st.session_state.get("selected_claims", [])])
                tracker_claims = set([])
                if tracker is not None and not tracker.empty and "claim_id" in tracker.columns:
                    tracker_claims = set(tracker["claim_id"].astype(str).str.strip().unique())

                in_triage_not_selected = sorted(list(triage_claims - session_claims))
                selected_not_in_triage = sorted(list(session_claims - triage_claims))
                in_tracker_not_in_triage = sorted(list(tracker_claims - triage_claims))
                selected_not_in_tracker = sorted(list(session_claims - tracker_claims))

                st.write(f"Triage flagged claims: {len(triage_claims):,}")
                st.write(f"Session selected claims: {len(session_claims):,}")
                st.write(f"Tracker claims: {len(tracker_claims):,}")

                def _preview(name, seq):
                    if not seq:
                        st.write(f"{name}: 0")
                        return
                    st.write(f"{name}: {len(seq):,} — sample (up to 20):")
                    st.write(seq[:20])

                _preview("In triage but not selected", in_triage_not_selected)
                _preview("Selected but not in triage", selected_not_in_triage)
                _preview("In tracker but not in triage", in_tracker_not_in_triage)
                _preview("Selected but not in tracker", selected_not_in_tracker)

                # show example rows for the first few missing IDs (helps debug normalization)
                if selected_not_in_triage:
                    st.markdown("**Example: Selected but not in triage — show staged rows for first 10**")
                    probe = list(selected_not_in_triage)[:10]
                    # show any matching rows from raw, features, or tracker
                    raw_rows = df_raw[df_raw.apply(lambda r: str(r.get("claim_id", "")).strip() in probe, axis=1)]
                    if not raw_rows.empty:
                        st.write("Matching rows from staged raw CSV:")
                        st.dataframe(raw_rows.head(10))
                    else:
                        st.write("No matching rows in staged raw CSV for those claim_ids.")
                    feat_rows = df_feat[df_feat["claim_id"].astype(str).isin(probe)]
                    if not feat_rows.empty:
                        st.write("Matching rows from processed features:")
                        st.dataframe(feat_rows.head(10))
                    tr_rows = tracker[tracker["claim_id"].astype(str).isin(probe)] if (tracker is not None and not tracker.empty) else pd.DataFrame()
                    if not tr_rows.empty:
                        st.write("Matching rows in tracker:")
                        st.dataframe(tr_rows.head(10))
            except Exception as e:
                st.write("Diagnostics failed:", str(e))


def main():
    st.sidebar.title("Navigation")
    # give the sidebar radio a session_state key so we can programmatically navigate
    # allow programmatic navigation via a query param 'page' or a one-shot goto flag stored in session_state['goto_page']
    options = ["Home", "Upload Claims", "Overview", "Filter & Triage", "Export Report"]
    default_page = None
    # prefer explicit query param if present (use read-only st.query_params)
    try:
        params = st.query_params
    except Exception:
        try:
            params = st.experimental_get_query_params()
        except Exception:
            params = {}
    if params and params.get("page"):
        candidate = params.get("page")[0]
        # consume the param so it doesn't persist
        try:
            st.experimental_set_query_params()
        except Exception:
            pass
        default_page = candidate
    elif "goto_page" in st.session_state:
        default_page = st.session_state.pop("goto_page")

    if default_page and default_page in options:
        default_index = options.index(default_page)
    else:
        default_index = 0
    page = st.sidebar.radio("Go to", options, index=default_index)
    if page == "Home":
        home_page()
    elif page == "Upload Claims":
        upload_page()
    elif page == "Overview":
        overview_page()
    elif page == "Filter & Triage":
        filter_triage_page()
    elif page == "Export Report":
        selected_claims_page()


if __name__ == "__main__":
    main()

