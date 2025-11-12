
from typing import Dict, Optional
import pandas as pd

class MetricSet:
    def __init__(
        self,
        per_class: Optional[Dict[str, Dict[str, float]]] = None,
        macro_avg: Optional[float] = None,
        weighted_avg: Optional[float] = None,
        accuracy: Optional[float] = None,
        confusion_matrix: Optional[Dict[str, Dict[str, int]]] = None,
    ):
        self.per_class = per_class or {}
        self.macro_avg = macro_avg
        self.weighted_avg = weighted_avg
        self.accuracy = accuracy
        self.confusion_matrix = confusion_matrix

    def to_dataframe(self) -> pd.DataFrame:
        """
        Returns a DataFrame with per-class precision, recall, and F1.
        """
        return pd.DataFrame.from_dict(self.per_class, orient="index")

    def highlight_class(self, label: str) -> Dict[str, float]:
        """
        Returns metrics for a specific class label.
        """
        return self.per_class.get(label, {})

    def compare(self, other: "MetricSet") -> Dict[str, float]:
        """
        Returns the delta in macro F1, weighted F1, and accuracy.
        """
        return {
            "delta_macro_f1": (self.macro_avg or 0) - (other.macro_avg or 0),
            "delta_weighted_f1": (self.weighted_avg or 0) - (other.weighted_avg or 0),
            "delta_accuracy": (self.accuracy or 0) - (other.accuracy or 0),
        }

    def export_summary(self) -> Dict[str, float]:
        """
        Returns a compact summary for sweep-level reporting.
        """
        return {
            "macro_f1": self.macro_avg,
            "weighted_f1": self.weighted_avg,
            "accuracy": self.accuracy,
        }
