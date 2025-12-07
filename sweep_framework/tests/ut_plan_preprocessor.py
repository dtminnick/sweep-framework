
import pandas as pd
from io import StringIO

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from sweep_framework.data.plan_preprocessor import PlanPreprocessor

# Load your CSV file
df = pd.read_csv("sweep_framework/documentation/usage/raw_input_data.txt")

# Group by plan_id
plan_id = 102345
plan_data = {
    "plan_id": str(plan_id),
    "months": df[df["plan_id"] == plan_id].to_dict(orient="records"),
    "static": {
        "hardship_allowed": 1,
        "loan_allowed": 1,
        "participants": 250,
        "median_balance": 12000,
        "median_age": 45
    },
    "label": 1
}

# Run with verbose output
preproc = PlanPreprocessor(config_path="sweep_framework/config/features.yml", verbose=True)
dynamic_seq, static_vec, label = preproc.preprocess_plan(plan_data)
