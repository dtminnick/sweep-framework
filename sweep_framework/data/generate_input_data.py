
import pandas as pd
import numpy as np

# Seed for reproducibility
rng = np.random.default_rng(42)

# Base year months
months = pd.date_range("2024-01-01", "2024-12-31", freq="MS").strftime("%Y-%m").tolist()

# Two base seasonal patterns (from your sample) for label=1 and label=0
def base_series_label1(m):
    # termination_req, termination_comp, termination_days
    tr_req = [12,8,10,9,11,7,13,9,10,8,12,9][m]
    tr_comp = [11,8,9,9,10,7,12,9,9,8,11,9][m]
    tr_days = [3.5,3.0,3.2,3.0,3.4,2.9,3.6,3.1,3.3,3.0,3.5,3.2][m]
    # in-service
    is_req  = [2,1,2,1,2,1,2,1,2,1,2,1][m]
    is_comp = [2,1,2,1,2,1,2,1,2,1,2,1][m]
    is_days = [2.0,2.5,2.1,2.3,2.2,2.4,2.0,2.5,2.2,2.4,2.1,2.3][m]
    # rollover
    ro_req  = [1,2,1,2,1,2,1,2,1,2,1,2][m]
    ro_comp = [1,2,1,2,1,2,1,2,1,2,1,2][m]
    ro_days = [4.0,3.8,4.1,3.9,4.2,3.7,4.3,3.8,4.0,3.9,4.1,3.8][m]
    # hardship
    hs_req  = [0,1,0,1,0,1,0,1,0,1,0,1][m]
    hs_comp = [0,1,0,1,0,1,0,1,0,1,0,1][m]
    hs_days = [0.0,4.2,0.0,4.0,0.0,4.1,0.0,4.3,0.0,4.2,0.0,4.4][m]
    # loan
    ln_req  = [3,4,2,3,4,3,5,4,3,2,4,3][m]
    ln_comp = [3,4,2,3,4,3,5,4,3,2,4,3][m]
    ln_days = [5.0,6.0,5.5,5.8,6.2,5.9,6.5,6.1,5.7,5.6,6.3,5.9][m]
    # exceptions + sla
    ex_open   = [2,1,1,2,1,2,2,1,2,1,2,1][m]
    ex_resolv = [2,1,1,2,1,2,2,1,2,1,2,1][m]
    sla = [0.05,0.02,0.03,0.04,0.05,0.03,0.06,0.04,0.05,0.03,0.05,0.04][m]
    return dict(
        termination_req=tr_req, termination_comp=tr_comp, termination_days=tr_days,
        inserv_req=is_req, inserv_comp=is_comp, inserv_days=is_days,
        rollover_req=ro_req, rollover_comp=ro_comp, rollover_days=ro_days,
        hardship_req=hs_req, hardship_comp=hs_comp, hardship_days=hs_days,
        loan_req=ln_req, loan_comp=ln_comp, loan_days=ln_days,
        exceptions_opened=ex_open, exceptions_resolved=ex_resolv, sla_breach_rate=sla
    )

def base_series_label0(m):
    tr_req = [5,7,6,8,5,7,6,8,5,7,6,7][m]
    tr_comp = [4,6,5,7,4,6,5,7,4,6,5,6][m]
    tr_days = [6.0,5.5,5.8,5.6,6.2,5.7,5.9,5.5,6.1,5.6,5.8,5.7][m]
    is_req  = [0,1,0,1,0,1,0,1,0,1,0,1][m]
    is_comp = [0,1,0,1,0,1,0,1,0,1,0,1][m]
    is_days = [0.0,3.0,0.0,2.8,0.0,2.9,0.0,2.7,0.0,2.9,0.0,2.8][m]
    ro_req  = [3,1,2,1,3,1,2,1,3,1,2,1][m]
    ro_comp = [2,1,2,1,2,1,2,1,2,1,2,1][m]
    ro_days = [7.0,6.0,6.5,6.2,7.1,6.3,6.6,6.1,7.2,6.4,6.7,6.2][m]
    hs_req  = [2,0,1,0,2,0,1,0,2,0,1,0][m]
    hs_comp = [1,0,1,0,1,0,1,0,1,0,1,0][m]
    hs_days = [8.0,0.0,7.5,0.0,8.2,0.0,7.8,0.0,8.1,0.0,7.9,0.0][m]
    ln_req  = [6,5,4,6,5,4,6,5,4,6,5,4][m]
    ln_comp = [5,5,4,6,5,4,6,5,4,6,5,4][m]
    ln_days = [10.0,9.0,9.5,9.8,10.2,9.4,10.1,9.7,9.9,10.0,9.6,9.8][m]
    ex_open   = [4,2,3,2,4,2,3,2,4,2,3,2][m]
    ex_resolv = [3,2,3,2,3,2,3,2,3,2,3,2][m]
    sla = [0.15,0.10,0.12,0.11,0.14,0.13,0.12,0.11,0.13,0.12,0.14,0.12][m]
    return dict(
        termination_req=tr_req, termination_comp=tr_comp, termination_days=tr_days,
        inserv_req=is_req, inserv_comp=is_comp, inserv_days=is_days,
        rollover_req=ro_req, rollover_comp=ro_comp, rollover_days=ro_days,
        hardship_req=hs_req, hardship_comp=hs_comp, hardship_days=hs_days,
        loan_req=ln_req, loan_comp=ln_comp, loan_days=ln_days,
        exceptions_opened=ex_open, exceptions_resolved=ex_resolv, sla_breach_rate=sla
    )

# Static templates for the two base profiles
static_label1 = {
    "hardship_allowed": 1, "loan_allowed": 1, "inserv_allowed": 1,
    "participants": 250, "median_balance": 12000, "median_age": 45,
    "fee_distribution": 25.0, "fee_loan_origination": 50.0,
    "vendor_type": "custodian_A", "fee_tier": "low"
}
static_label0 = {
    "hardship_allowed": 1, "loan_allowed": 1, "inserv_allowed": 0,
    "participants": 120, "median_balance": 8000, "median_age": 52,
    "fee_distribution": 30.0, "fee_loan_origination": 75.0,
    "vendor_type": "custodian_B", "fee_tier": "high"
}

# Expand to 12 plans: 6 of label=1 and 6 of label=0
plan_ids_label1 = [102345, 102346, 102347, 102348, 102349, 102350]
plan_ids_label0 = [204567, 204568, 204569, 204570, 204571, 204572]

rows = []
for pid in plan_ids_label1:
    # slight per-plan jitter to avoid identical values
    jitter = rng.normal(0, 0.1)
    for mi, month in enumerate(months):
        s = base_series_label1(mi)
        # apply small jitter to continuous-day metrics and rates
        s["termination_days"] += jitter
        s["inserv_days"] += jitter
        s["rollover_days"] += jitter
        s["hardship_days"] = max(0.0, s["hardship_days"] + jitter)
        s["loan_days"] += jitter
        s["sla_breach_rate"] = max(0.0, min(1.0, s["sla_breach_rate"] + jitter/10))
        row = {
            "plan_id": pid, "month": month, **s,
            **static_label1, "label": 1
        }
        rows.append(row)

# Vary vendor_type/fee_tier across label=1 plans to exercise embeddings
vendor_cycle1 = ["custodian_A", "custodian_C", "custodian_A", "custodian_D", "custodian_C", "custodian_A"]
fee_cycle1 = ["low", "medium", "low", "high", "medium", "low"]
for i, pid in enumerate(plan_ids_label1):
    for r in rows:
        if r["plan_id"] == pid:
            r["vendor_type"] = vendor_cycle1[i]
            r["fee_tier"] = fee_cycle1[i]

for pid in plan_ids_label0:
    jitter = rng.normal(0, 0.1)
    for mi, month in enumerate(months):
        s = base_series_label0(mi)
        s["termination_days"] += jitter
        s["inserv_days"] += jitter
        s["rollover_days"] += jitter
        s["hardship_days"] = max(0.0, s["hardship_days"] + jitter)
        s["loan_days"] += jitter
        s["sla_breach_rate"] = max(0.0, min(1.0, s["sla_breach_rate"] + jitter/10))
        row = {
            "plan_id": pid, "month": month, **s,
            **static_label0, "label": 0
        }
        rows.append(row)

# Vary vendor_type/fee_tier across label=0 plans
vendor_cycle0 = ["custodian_B", "custodian_D", "custodian_B", "custodian_C", "custodian_D", "custodian_B"]
fee_cycle0 = ["high", "medium", "high", "low", "medium", "high"]
for i, pid in enumerate(plan_ids_label0):
    for r in rows:
        if r["plan_id"] == pid:
            r["vendor_type"] = vendor_cycle0[i]
            r["fee_tier"] = fee_cycle0[i]

# Build DataFrame
df_expanded = pd.DataFrame(rows)

# Optional: ensure integer types where appropriate
int_cols = [
    "plan_id","termination_req","termination_comp","inserv_req","inserv_comp",
    "rollover_req","rollover_comp","hardship_req","hardship_comp",
    "loan_req","loan_comp","exceptions_opened","exceptions_resolved",
    "hardship_allowed","loan_allowed","inserv_allowed","participants","label"
]
for c in int_cols:
    df_expanded[c] = df_expanded[c].astype(int)

# Save to CSV
df_expanded.to_csv("sweep_framework/documentation/usage/raw_input_data_expanded.txt", index=False)

print("Expanded dataset written with", df_expanded["plan_id"].nunique(), "plans and", len(df_expanded), "rows.")
