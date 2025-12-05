
# sweep_framework/model/model_run.py
"""
ModelRun orchestrates training, evaluation, and reporting for a single experiment.

Responsibilities:
- Initialize model, optimizer, scheduler, and loss strategy from ModelConfig.
- Train the model with early stopping.
- Evaluate on validation and test sets.
- Track metrics across epochs.
- Export summaries for sweep-level reporting.
"""

from typing import Optional
import torch
from sweep_framework.config.model_config import ModelConfig
from sweep_framework.data.dataset import Dataset
from sweep_framework.metrics.metric_set import MetricSet
from sweep_framework.model.loss_strategy import LossStrategy


class ModelRun:
    """
    Encapsulates a single training run.

    Attributes:
        config (ModelConfig): Configuration for the run.
        dataset (Dataset): Dataset with train/val/test loaders.
        loss_strategy (LossStrategy): Strategy for computing loss.
        model (nn.Module): PyTorch model.
        optimizer (Optimizer): Optimizer instance.
        scheduler (LRScheduler): Learning rate scheduler.
        device (torch.device): Training device (CPU/GPU).
        train_metrics, val_metrics, test_metrics (MetricSet): Metrics for splits.
        epoch_logs (list): Log of metrics per epoch.
        early_stopping_triggered (bool): Flag for early stopping.
    """

    def __init__(self, config: ModelConfig, dataset: Dataset, loss_strategy: Optional[LossStrategy] = None):
        self.config = config
        self.dataset = dataset
        self.loss_strategy = loss_strategy or LossStrategy.from_config(config)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = config.build_model()
        self.model.to(self.device)
        self.optimizer = config.build_optimizer(self.model.parameters())
        self.scheduler = config.build_scheduler(self.optimizer)

        self.train_metrics: Optional[MetricSet] = None
        self.val_metrics: Optional[MetricSet] = None
        self.test_metrics: Optional[MetricSet] = None
        self.epoch_logs = []
        self.early_stopping_triggered = False

    def _iterate_loader(self, loader, train_mode: bool) -> MetricSet:
        """
        Run one pass over a DataLoader.

        Args:
            loader (DataLoader): PyTorch DataLoader.
            train_mode (bool): If True, perform training; else evaluation.

        Returns:
            MetricSet: Metrics for the split.
        """
        metrics = MetricSet()
        self.model.train(train_mode)

        if train_mode:
            for batch in loader:
                inputs = batch["input"].to(self.device)
                targets = batch["target"].to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_strategy.compute(outputs, targets)
                loss.backward()
                if self.config.clip_grad_norm:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip_grad_norm)
                self.optimizer.step()
                metrics.update(outputs, targets)
        else:
            with torch.no_grad():
                for batch in loader:
                    inputs = batch["input"].to(self.device)
                    targets = batch["target"].to(self.device)
                    outputs = self.model(inputs)
                    metrics.update(outputs, targets)

        metrics.finalize()
        return metrics

    def train(self):
        """
        Train the model with early stopping based on validation macro F1.
        """
        best_val_score = float("-inf")
        patience_counter = 0

        for epoch in range(self.config.num_epochs):
            train_metrics = self._iterate_loader(self.dataset.train_loader, train_mode=True)
            val_metrics = self._iterate_loader(self.dataset.val_loader, train_mode=False)

            # Step scheduler (simple StepLR by default)
            self.scheduler.step()

            self.epoch_logs.append({
                "epoch": epoch,
                "train_macro_f1": train_metrics.macro_avg,
                "val_macro_f1": val_metrics.macro_avg,
            })

            # Early stopping
            if (val_metrics.macro_avg or -1) > best_val_score:
                best_val_score = val_metrics.macro_avg or -1
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config.patience:
                    self.early_stopping_triggered = True
                    break

            self.train_metrics = train_metrics
            self.val_metrics = val_metrics

    def evaluate(self, split: str) -> MetricSet:
        """
        Evaluate the model on a given split.

        Args:
            split (str): "val" or "test".

        Returns:
            MetricSet: Metrics for the split.
        """
        loader = {"val": self.dataset.val_loader, "test": self.dataset.test_loader}.get(split)
        if loader is None:
            raise ValueError(f"Invalid split: {split}")
        return self._iterate_loader(loader, train_mode=False)

    def run(self):
        """
        Run full training and evaluation cycle.
        """
        self.train()
        self.val_metrics = self.evaluate("val")
        self.test_metrics = self.evaluate("test")

    def export_summary(self) -> dict:
        """
        Export summary for reporting.

        Returns:
            dict: Config, metrics, run metadata.
        """
        return {
            "config": self.config.to_dict(),
            "macro_f1": (self.test_metrics.macro_avg if self.test_metrics else None),
            "weighted_f1": (self.test_metrics.weighted_avg if self.test_metrics else None),
            "accuracy": (self.test_metrics.accuracy if self.test_metrics else None),
            "val_macro_f1": (self.val_metrics.macro_avg if self.val_metrics else None),
            "run_group": self.config.run_group,
            "early_stopping": self.early_stopping_triggered,
        }
