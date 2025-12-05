
# sweep_framework/analysis/sweep/sweep_analyzer.py
"""
SweepAnalyzer provides tools for analyzing results across multiple sweeps.

Responsibilities:
- Collect results from a SweepRegistry.
- Rank runs by a chosen metric.
- Compare groups of sweeps.
- Summarize trends across hyperparameters.
- Export results as DataFrames for further analysis.
"""

import pandas as pd
from typing import List, Dict
from sweep_framework.sweep.sweep_registry import SweepRegistry


class SweepAnalyzer:
    """
    Analyze results from multiple sweeps.

    Attributes:
        registry (SweepRegistry): Registry containing sweeps.
        results (List[Dict]): Flattened list of run summaries across sweeps.
    """

    def __init__(self, registry: SweepRegistry):
        self.registry = registry
        self.results: List[Dict] = self._collect_results()

    def _collect_results(self) -> List[Dict]:
        """Collect summaries from all sweeps in the registry."""
        return [summary for sweep in self.registry.sweeps.values() for summary in sweep.summarize()]

    def rank(self, metric: str = "macro_f1") -> pd.DataFrame:
        """
        Rank runs by a given metric.

        Args:
            metric (str): Metric to sort by.

        Returns:
            DataFrame: Runs sorted by metric (descending).
        """
        df = pd.DataFrame(self.results)
        return df.sort_values(by=metric, ascending=False)

    def compare_groups(self, group_a: str, group_b: str, metric: str = "macro_f1") -> pd.DataFrame:
        """
        Compare two groups of sweeps.

        Args:
            group_a (str): First group name.
            group_b (str): Second group name.
            metric (str): Metric to sort by.

        Returns:
            DataFrame: Combined results for both groups.
        """
        sweeps_a = self.registry.get_group(group_a)
        sweeps_b = self.registry.get_group(group_b)

        def collect(sweeps):
            return [summary for sweep in sweeps for summary in sweep.summarize()]

        df_a = pd.DataFrame(collect(sweeps_a))
        df_b = pd.DataFrame(collect(sweeps_b))
        df_a["group"] = group_a
        df_b["group"] = group_b
        return pd.concat([df_a, df_b]).sort_values(by=metric, ascending=False)

    def summarize_trends(self, param: str, metric: str = "macro_f1") -> pd.DataFrame:
        """
        Summarize trends across a hyperparameter.

        Args:
            param (str): Parameter name (e.g., "dropout").
            metric (str): Metric to average.

        Returns:
            DataFrame: Mean metric per parameter value.
        """
        df = pd.DataFrame(self.results)
        return df.groupby(param)[metric].mean().reset_index().sort_values(by=metric, ascending=False)

    def to_dataframe(self) -> pd.DataFrame:
        """
        Export all results as a DataFrame.

        Returns:
            DataFrame: Flattened results.
        """
        return pd.DataFrame(self.results)
