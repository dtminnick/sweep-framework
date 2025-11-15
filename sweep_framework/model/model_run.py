
from typing import Optional
from sweep_framework.config.model_config import ModelConfig
from sweep_framework.data.dataset import Dataset
from sweep_framework.metrics.metric_set import MetricSet
from sweep_framework.model.loss_strategy import LossStrategy
import torch

class ModelRun:
    def __init__(
        self,
        config: ModelConfig,
        dataset: Dataset,
        loss_strategy: Optional[LossStrategy] = None,
    ):
        self.config = config
        self.dataset = dataset
        self.loss_strategy = loss_strategy or LossStrategy(
            "focal" if config.use_focal_loss else config.criterion
        )
        self.model = config.build_model()
        self.optimizer = config.build_optimizer(self.model.parameters())
        self.scheduler = config.build_scheduler(self.optimizer)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.train_metrics = None
        self.val_metrics = None
        self.test_metrics = None
        self.epoch_logs = []
        self.early_stopping_triggered = False

    def train(self):
        best_val_score = float("-inf")
        patience_counter = 0

        for epoch in range(self.config.num_epochs):
            self.model.train()
            train_metrics = MetricSet()

            for batch in self.dataset.train_loader:
                inputs, targets = batch["input"].to(self.device), batch["target"].to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_strategy.compute(outputs, targets)
                loss.backward()
                self.optimizer.step()

                train_metrics.update(outputs, targets)

            self.scheduler.step()
            val_metrics = self.evaluate("val")
            self.epoch_logs.append({
                "epoch": epoch,
                "train_macro_f1": train_metrics.macro_avg,
                "val_macro_f1": val_metrics.macro_avg,
            })

            # Early stopping
            if val_metrics.macro_avg > best_val_score:
                best_val_score = val_metrics.macro_avg
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config.patience:
                    self.early_stopping_triggered = True
                    break

            self.train_metrics = train_metrics

    def evaluate(self, split: str) -> MetricSet:
        self.model.eval()
        metrics = MetricSet()
        loader = {
            "val": self.dataset.val_loader,
            "test": self.dataset.test_loader
        }.get(split)

        if loader is None:
            raise ValueError(f"Invalid split: {split}")

        with torch.no_grad():
            for batch in loader:
                inputs, targets = batch["input"].to(self.device), batch["target"].to(self.device)
                outputs = self.model(inputs)
                metrics.update(outputs, targets)

        return metrics
    
    def run(self):
        self.train()
        self.val_metrics = self.evaluate("val")
        self.test_metrics = self.evaluate("test")

    def export_summary(self) -> dict:
        return {
            "config": self.config.to_dict(),
            "val_macro_f1": self.val_metrics.macro_avg if self.val_metrics else None,
            "test_macro_f1": self.test_metrics.macro_avg if self.test_metrics else None,
        }
    