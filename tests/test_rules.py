import pandas as pd
from pi_core.rules import run_rules

def test_run_rules_flags():
    # create a small dataframe with rows to trigger each rule
    data = [
        # duplicate (same member/provider/date/cpt)
    {"claim_id": "dup1", "member_id": "m1", "provider_id": "p1", "service_date": "2020-01-01", "cpt_code": "1000", "paid_amount": 100.0, "allowed_amount": 90.0, "median_allowed": 95.0, "units": 1, "insurance_type": "Commercial", "geo": "00000"},
    {"claim_id": "dup2", "member_id": "m1", "provider_id": "p1", "service_date": "2020-01-01", "cpt_code": "1000", "paid_amount": 110.0, "allowed_amount": 90.0, "median_allowed": 95.0, "units": 1, "insurance_type": "Commercial", "geo": "00000"},
        # overpay
    {"claim_id": "ov1", "member_id": "m2", "provider_id": "p2", "service_date": "2020-02-01", "cpt_code": "2000", "paid_amount": 210.0, "allowed_amount": 100.0, "median_allowed": 105.0, "units": 1, "insurance_type": "Commercial", "geo": "11111"},
        # price outlier
    {"claim_id": "pr1", "member_id": "m3", "provider_id": "p3", "service_date": "2020-03-01", "cpt_code": "3000", "paid_amount": 300.0, "allowed_amount": 150.0, "median_allowed": 100.0, "units": 1, "insurance_type": "Commercial", "geo": "22222"},
        # low units, high charge
    {"claim_id": "lu1", "member_id": "m4", "provider_id": "p4", "service_date": "2020-04-01", "cpt_code": "4000", "paid_amount": 900.0, "allowed_amount": 200.0, "median_allowed": 200.0, "units": 1, "insurance_type": "Commercial", "geo": "33333"},
        # medicare variance
    {"claim_id": "mc1", "member_id": "m5", "provider_id": "p5", "service_date": "2020-05-01", "cpt_code": "5000", "paid_amount": 150.0, "allowed_amount": 100.0, "median_allowed": 100.0, "units": 1, "insurance_type": "Medicare", "paid_to_allowed": 1.6, "geo": "44444"},
    ]

    df = pd.DataFrame(data)
    flags = run_rules(df)

    # sanity: claim ids flagged should include our test ids
    flagged_ids = set(flags["claim_id"].unique())
    assert {"dup1", "dup2", "ov1", "pr1", "lu1", "mc1"}.issubset(flagged_ids)

    # check at least one expected flag type exists
    types = set(flags["flag_type"].unique())
    expected = {"RULE_DUPLICATE", "RULE_OVERPAY", "RULE_PRICE_OUTLIER", "RULE_LOW_UNITS_HIGH_CHARGE", "RULE_MEDICARE_VARIANCE"}
    assert expected.issubset(types)
