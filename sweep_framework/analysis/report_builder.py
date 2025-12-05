
# sweep_framework/report/report_builder.py
"""
ReportBuilder generates human-readable summaries and trend insights from sweeps.

Responsibilities:
- Build summary tables of top runs.
- Summarize trends across hyperparameters.
- Annotate runs with notes.
- Export results in structured format for reporting.
"""

import pandas as pd
from sweep_framework.sweep.sweep_analyzer import SweepAnalyzer


class ReportBuilder:
    """
    Build reports from sweep analysis.

    Attributes:
        analyzer (SweepAnalyzer): Analyzer providing results.
        notes (dict): User annotations keyed by run group.
    """

    def __init__(self, analyzer: SweepAnalyzer):
        self.analyzer = analyzer
        self.notes = {}

    def build_summary(self, metric: str = "macro_f1", top_k: int = 5) -> pd.DataFrame:
        """
        Build summary table of top runs.

        Args:
            metric (str): Metric to rank by.
            top_k (int): Number of top runs to include.

        Returns:
            DataFrame: Top runs.
        """
        df = self.analyzer.rank(metric=metric)
        return df.head(top_k)

    def build_trends(self, param: str, metric: str = "macro_f1") -> pd.DataFrame:
        """
        Summarize trends across a hyperparameter.

        Args:
            param (str): Parameter name.
            metric (str): Metric to average.

        Returns:
            DataFrame: Mean metric per parameter value.
        """
        return self.analyzer.summarize_trends(param=param, metric=metric)

    def annotate(self, summary: dict, note: str):
        """
        Add a note to a run summary.

        Args:
            summary (dict): Run summary.
            note (str): Annotation text.
        """
        config_id = summary.get("config", {}).get("run_group", "unknown")
        self.notes[config_id] = note

    def export(self, format: str = "dict") -> dict:
        """
        Export report as a dictionary.

        Args:
            format (str): Output format ("dict").

        Returns:
            dict: Report with summary, trends, and annotations.
        """
        df = self.analyzer.to_dataframe()
        candidate_params = ["dropout", "learning_rate", "hidden_dim", "num_layers", "bidirectional"]
        trend_insights = {}
        for param in candidate_params:
            try:
                trend_insights[param] = self.build_trends(param).to_dict(orient="records")
            except Exception:
                pass
        return {
            "summary_table": self.build_summary().to_dict(orient="records"),
            "trend_insights": trend_insights,
            "annotations": self.notes,
        }
