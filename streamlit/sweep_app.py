
# streamlit/sweep_app.py

import streamlit as st
from sweep_framework.registry import SweepRegistry
from sweep_framework.analysis.sweep_analyzer import SweepAnalyzer
from sweep_framework.analysis.report_builder import ReportBuilder
import pandas as pd

st.set_page_config(page_title="Sweep Dashboard", layout="wide")

st.title("🔍 Sweep Explorer")

# Load registry and analyzer
registry = SweepRegistry()
analyzer = SweepAnalyzer(registry)
report = ReportBuilder(analyzer)

# Sidebar controls
sweep_names = registry.list_all()
selected_sweep = st.sidebar.selectbox("Select Sweep", sweep_names)
metric = st.sidebar.selectbox("Metric to rank by", ["macro_f1", "accuracy", "weighted_f1"])
top_k = st.sidebar.slider("Top K configs", 1, 10, 5)

# Summary table
st.subheader(f"Top {top_k} configs in '{selected_sweep}' ranked by {metric}")
summary_df = report.build_summary(metric=metric, top_k=top_k)
st.dataframe(summary_df)

# Trend analysis
param = st.selectbox("Parameter to analyze", ["dropout", "learning_rate", "hidden_dim"])
trend_df = report.build_trends(param=param, metric=metric)

st.subheader(f"📈 Trend: {param} vs {metric}")
st.bar_chart(trend_df.set_index(param))

# Optional group comparison
group_names = list(registry.groups.keys())
group_a = st.selectbox("Compare Group A", group_names)
group_b = st.selectbox("Compare Group B", group_names)

if group_a != group_b:
    st.subheader(f"📊 Comparing '{group_a}' vs '{group_b}' on {metric}")
    comparison_df = analyzer.compare_groups(group_a, group_b, metric=metric)
    st.dataframe(comparison_df)
