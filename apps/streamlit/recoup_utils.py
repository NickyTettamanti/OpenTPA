# apps/streamlit/recoup_utils.py
import os, hashlib, random, datetime as dt
import pandas as pd
from jinja2 import Environment, BaseLoader

TRACKER_PATH = os.path.join("artifacts", "recovery_tracker.csv")

# ---------------- Provider directory (deterministic fake) ----------------
def _seed_from(provider_id: str) -> int:
    h = hashlib.sha256(str(provider_id).encode()).hexdigest()
    return int(h[:8], 16)

def provider_contact(provider_id: str, provider_name: str | None = None) -> dict:
    """Return stable fake contact info for a provider_id (deterministic by hash)."""
    r = random.Random(_seed_from(provider_id))
    names = ["Medical Group", "Internal Medicine", "Family Practice", "Clinic", "Health LLC", "Primary Care"]
    streets = ["Broadway", "Main St", "1st Ave", "Maple Rd", "Market St", "Pine Ave"]
    cities = ["New York", "Newark", "Jersey City", "Brooklyn", "Queens", "Hoboken"]
    states = ["NY", "NJ"]
    zips = [str(10000 + r.randint(0, 89999))[:5] for _ in range(6)]
    prov_name = provider_name or f"Provider {provider_id}"
    return {
        "provider_name": prov_name,
        "address1": f"{r.randint(10, 9999)} {r.choice(streets)}",
        "address2": "",
        "city": r.choice(cities),
        "state": r.choice(states),
        "zip": r.choice(zips),
        "fax": f"({r.randint(200, 989)}) {r.randint(200, 989)}-{r.randint(1000, 9999)}",
        "phone": f"({r.randint(200, 989)}) {r.randint(200, 989)}-{r.randint(1000, 9999)}",
        "email": f"billing@prov{str(provider_id)[-4:]}.example.com",
    }

# ---------------- Recovery tracker persistence ----------------
def load_tracker() -> pd.DataFrame:
    if os.path.exists(TRACKER_PATH):
        df = pd.read_csv(TRACKER_PATH, parse_dates=["sent_date","recoup_date"], dtype={"claim_id":str})
    else:
        os.makedirs(os.path.dirname(TRACKER_PATH), exist_ok=True)
        df = pd.DataFrame(columns=[
            "claim_id","provider_id","flag_types","est_savings",
            "sent","sent_date","channel","letter_path",
            "recouped","recoup_amount","recoup_date","notes"
        ])
    return df

def save_tracker(df: pd.DataFrame) -> None:
    os.makedirs(os.path.dirname(TRACKER_PATH), exist_ok=True)
    df.to_csv(TRACKER_PATH, index=False)

def upsert_tracker(row: dict) -> pd.DataFrame:
    df = load_tracker()
    mask = (df["claim_id"] == row["claim_id"])
    if mask.any():
        for k,v in row.items():
            df.loc[mask, k] = v
    else:
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    save_tracker(df)
    return df

# ---------------- Letter generation ----------------
LETTER_TEMPLATE = """
{{ today }}
{{ payer_name }}
{{ payer_address1 }}
{{ payer_city }}, {{ payer_state }} {{ payer_zip }}

ATTN: Billing Department — {{ contact.provider_name }}
Fax: {{ contact.fax }} | Email: {{ contact.email }}

Subject: Notice of Overpayment and Request for Refund — Claim {{ claim.claim_id }}

Dear Billing Team,

We are writing regarding the claim listed below, identified by our payment integrity review as overpaid.

- Provider ID: {{ claim.provider_id }}
- Member ID: {{ claim.member_id }}
- Date of Service: {{ claim.service_date }}
- CPT Code: {{ claim.cpt_code }}
 - Paid Amount: ${{ "{:,.2f}".format(claim.paid_amount|default(0)) }}
 - Allowed Amount (reference): ${{ "{:,.2f}".format(claim.allowed_amount|default(0)) }}
 - Median Allowed (CPT×geo): ${{ "{:,.2f}".format(claim.median_allowed|default(0)) }}
- Reason: {{ reason }}

Requested Action:
Please remit an overpayment refund of **${{ "{:,.2f}".format(recoup_amount) }}** within 30 days of this notice, or contact us to coordinate an offset against future payments. If you dispute this finding, respond with supporting documentation.

Refunds by check may be sent to:
{{ payer_name }} — Overpayment Recovery
{{ payer_address1 }}
{{ payer_city }}, {{ payer_state }} {{ payer_zip }}

Sincerely,
Payment Integrity
{{ payer_name }}
"""

def render_letter_text(claim: dict, contact: dict, reason: str, recoup_amount: float, payer_profile: dict) -> str:
    env = Environment(loader=BaseLoader(), autoescape=False, trim_blocks=True, lstrip_blocks=True)
    tpl = env.from_string(LETTER_TEMPLATE)
    ctx = {
        "today": dt.date.today().strftime("%B %d, %Y"),
        "claim": claim,
        "contact": contact,
        "reason": reason,
        "recoup_amount": recoup_amount,
        **payer_profile
    }
    return tpl.render(**ctx)

def write_letter_file(text: str, claim_id: str) -> str:
    outdir = os.path.join("artifacts", "letters")
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"recoup_{claim_id}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path
