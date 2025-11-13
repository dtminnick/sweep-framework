
import pandas as pd
from typing import List, Dict
from sweep_framework.sweep.sweep_registry import SweepRegistry

class SweepAnalyzer:
    def __init__(self, registry: SweepRegistry):
        self.registry = registry
        self.results: List[Dict] = self._collect_results()

    def _collect_results(self) -> List[Dict]:
        return [
            summary
            for sweep in self.registry.sweeps.values()
            for summary in sweep.summarize()
        ]

    def rank(self, metric: str = "macro_f1") -> pd.DataFrame:
        df = pd.DataFrame(self.results)
        return df.sort_values(by=metric, ascending=False)

    def compare_groups(self, group_a: str, group_b: str, metric: str = "macro_f1") -> pd.DataFrame:
        sweeps_a = self.registry.get_group(group_a)
        sweeps_b = self.registry.get_group(group_b)

        def collect(sweeps):
            return [
                summary
                for sweep in sweeps
                for summary in sweep.summarize()
            ]

        df_a = pd.DataFrame(collect(sweeps_a))
        df_b = pd.DataFrame(collect(sweeps_b))

        df_a["group"] = group_a
        df_b["group"] = group_b

        return pd.concat([df_a, df_b]).sort_values(by=metric, ascending=False)

    def summarize_trends(self, param: str, metric: str = "macro_f1") -> pd.DataFrame:
        df = pd.DataFrame(self.results)
        return df.groupby(param)[metric].mean().reset_index().sort_values(by=metric, ascending=False)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.results)
