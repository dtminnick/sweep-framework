
import pandas as pd
from typing import Optional
from sweep_framework.analysis.sweep.sweep_analyzer import SweepAnalyzer

class ReportBuilder:
    def __init__(self, analyzer: SweepAnalyzer):
        self.analyzer = analyzer
        self.notes = {}

    def build_summary(self, metric: str = "macro_f1", top_k: int = 5) -> pd.DataFrame:
        df = self.analyzer.rank(metric=metric)
        return df.head(top_k)

    def build_trends(self, param: str, metric: str = "macro_f1") -> pd.DataFrame:
        return self.analyzer.summarize_trends(param=param, metric=metric)

    def annotate(self, summary: dict, note: str):
        config_id = summary.get("config", {}).get("run_group", "unknown")
        self.notes[config_id] = note

    def export(self, format: str = "dict") -> dict:
        return {
            "summary_table": self.build_summary().to_dict(orient="records"),
            "trend_insights": {
                param: self.build_trends(param).to_dict(orient="records")
                for param in ["dropout", "learning_rate", "hidden_dim"]
            },
            "annotations": self.notes,
        }
