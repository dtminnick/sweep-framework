
import pandas as pd
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from sweep_framework.data.plan_dataset import PlanDataset

# Load your CSV file
df = pd.read_csv("sweep_framework/documentation/usage/raw_input_data_expanded.txt")

# Build plan dictionaries for each unique plan_id
plans = []
for pid, group in df.groupby("plan_id"):
    months = group.to_dict(orient="records")

    static_fields = {
        "hardship_allowed": group.iloc[0]["hardship_allowed"],
        "loan_allowed": group.iloc[0]["loan_allowed"],
        "inserv_allowed": group.iloc[0]["inserv_allowed"],
        "participants": group.iloc[0]["participants"],
        "median_balance": group.iloc[0]["median_balance"],
        "median_age": group.iloc[0]["median_age"],
        "fee_distribution": group.iloc[0]["fee_distribution"],
        "fee_loan_origination": group.iloc[0]["fee_loan_origination"],
        "vendor_type": group.iloc[0]["vendor_type"],
        "fee_tier": group.iloc[0].get("fee_tier", "medium")
    }

    label = group.iloc[0]["label"]

    plans.append({
        "plan_id": str(pid),
        "months": months,
        "static": static_fields,
        "label": label
    })

# Instantiate PlanDataset (handles stats + preprocessing internally)
dataset = PlanDataset(plans, config_path="sweep_framework/config/features.yml", verbose=True)

# Inspect stats computed from training split
print("=== Normalization Stats (training only) ===")
print(dataset.stats)
print()
print()

# Inspect preprocessed examples
for dynamic_seq, static_vec, embedding_indices, static_embs, label in dataset.train_examples:
    print("Dynamic sequence shape:", dynamic_seq.shape)
    print("Static vector:", static_vec.tolist())
    print("Dynamic embedding indices:", {k: v.tolist() for k, v in embedding_indices.items()})
    print("Static embedding indices:", {k: v.item() for k, v in static_embs.items()})
    print("Label:", label)
    print()
    print()

