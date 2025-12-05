
# sweep_framework/sweep/sweep.py
"""
Sweep orchestrates multiple model runs over a configuration grid.

Responsibilities:
- Run all configurations on a dataset.
- Summarize results.
- Identify best model by metric.
- Export results for reporting.
"""

from typing import List, Optional
from sweep_framework.config.model_config import ModelConfig
from sweep_framework.data.dataset import Dataset
from sweep_framework.model.model_run import ModelRun


class Sweep:
    """
    Orchestrates multiple model runs.

    Attributes:
        name (str): Sweep name.
        config_grid (List[ModelConfig]): List of model configurations.
        dataset (Dataset): Dataset with train/val/test loaders.
        run_group (Optional[str]): Group identifier for runs.
        runs (List[ModelRun]): Completed runs.
    """

    def __init__(self, name: str, config_grid: List[ModelConfig], dataset: Dataset, run_group: Optional[str] = None):
        self.name = name
        self.config_grid = config_grid
        self.dataset = dataset
        self.run_group = run_group
        self.runs: List[ModelRun] = []

    def run_all(self):
        """
        Run all configurations in the grid.
        """
        for config in self.config_grid:
            config.run_group = self.run_group or self.name
            run = ModelRun(config=config, dataset=self.dataset)
            run.run()
            self.runs.append(run)

    def summarize(self) -> List[dict]:
        """
        Summarize all runs.

        Returns:
            List[dict]: Export summaries from each run.
        """
        return [run.export_summary() for run in self.runs]

    def get_best_model(self, metric: str = "val_macro_f1") -> Optional[ModelRun]:
        """
        Get the best model by a given metric.

        Args:
            metric (str): Metric to rank by.

        Returns:
            ModelRun: Best run, or None if no runs.
        """
        scored = [(run, run.export_summary().get(metric)) for run in self.runs if run.export_summary().get(metric) is not None]
        if not scored:
            return None
        return max(scored, key=lambda x: x[1])[0]

    def export_results(self) -> dict:
        """
        Export sweep-level results.

        Returns:
            dict: Sweep metadata and summaries.
        """
        return {
            "sweep_name": self.name,
            "run_group": self.run_group,
            "num_runs": len(self.runs),
            "summaries": self.summarize()
        }
