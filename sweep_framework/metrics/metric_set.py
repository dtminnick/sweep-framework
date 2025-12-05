
# sweep_framework/metrics/metric_set.py
"""
Metric tracking and reporting for classification tasks.

This module defines the MetricSet class, which:
- Collects predictions and labels across batches.
- Computes per-class precision, recall, F1, macro/weighted averages, and accuracy.
- Stores a confusion matrix.
- Provides utilities for comparison and export.
"""

from typing import Dict, Optional, List
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix


class MetricSet:
    """
    Container for classification metrics.

    Attributes:
        per_class (Dict[str, Dict[str, float]]): Per-class precision/recall/F1.
        macro_avg (Optional[float]): Macro-average F1 score.
        weighted_avg (Optional[float]): Weighted-average F1 score.
        accuracy (Optional[float]): Overall accuracy.
        confusion_matrix (Optional[List[List[int]]]): Confusion matrix as nested list.
        _preds (List[int]): Internal list of predictions.
        _labels (List[int]): Internal list of true labels.
    """

    def __init__(
        self,
        per_class: Optional[Dict[str, Dict[str, float]]] = None,
        macro_avg: Optional[float] = None,
        weighted_avg: Optional[float] = None,
        accuracy: Optional[float] = None,
        confusion_matrix_data: Optional[List[List[int]]] = None,
    ):
        self.per_class = per_class or {}
        self.macro_avg = macro_avg
        self.weighted_avg = weighted_avg
        self.accuracy = accuracy
        self.confusion_matrix = confusion_matrix_data
        self._preds: List[int] = []
        self._labels: List[int] = []

    def update(self, outputs: torch.Tensor, targets: torch.Tensor):
        """
        Add predictions and labels from a batch.

        Args:
            outputs (Tensor): Model outputs (batch_size x num_classes).
            targets (Tensor): True labels (batch_size).
        """
        preds = outputs.argmax(dim=1).detach().cpu().tolist()
        labels = targets.detach().cpu().tolist()
        self._preds.extend(preds)
        self._labels.extend(labels)

    def finalize(self):
        """
        Compute metrics after all batches have been processed.
        """
        if not self._labels:
            self.per_class, self.macro_avg, self.weighted_avg, self.accuracy = {}, None, None, None
            self.confusion_matrix = None
            return

        report = classification_report(self._labels, self._preds, output_dict=True, zero_division=0)
        self.per_class = {k: v for k, v in report.items() if k not in ["accuracy", "macro avg", "weighted avg"]}
        self.macro_avg = report["macro avg"]["f1-score"]
        self.weighted_avg = report["weighted avg"]["f1-score"]
        self.accuracy = report["accuracy"]
        self.confusion_matrix = confusion_matrix(self._labels, self._preds).tolist()

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert per-class metrics to a pandas DataFrame.

        Returns:
            DataFrame: Rows = classes, columns = precision/recall/F1.
        """
        return pd.DataFrame.from_dict(self.per_class, orient="index")

    def highlight_class(self, label: str) -> Dict[str, float]:
        """
        Get metrics for a specific class.

        Args:
            label (str): Class label.

        Returns:
            Dict[str, float]: Precision/recall/F1 for the class.
        """
        return self.per_class.get(label, {})

    def compare(self, other: "MetricSet") -> Dict[str, float]:
        """
        Compare metrics with another MetricSet.

        Args:
            other (MetricSet): Another MetricSet.

        Returns:
            Dict[str, float]: Deltas for macro F1, weighted F1, and accuracy.
        """
        return {
            "delta_macro_f1": (self.macro_avg or 0) - (other.macro_avg or 0),
            "delta_weighted_f1": (self.weighted_avg or 0) - (other.weighted_avg or 0),
            "delta_accuracy": (self.accuracy or 0) - (other.accuracy or 0),
        }

    def export_summary(self) -> Dict[str, float]:
        """
        Export compact summary for reporting.

        Returns:
            Dict[str, float]: Macro F1, weighted F1, accuracy.
        """
        return {
            "macro_f1": self.macro_avg,
            "weighted_f1": self.weighted_avg,
            "accuracy": self.accuracy,
        }
