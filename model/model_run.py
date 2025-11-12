
from typing import Optional
from sweep_framework.config.model_config import ModelConfig
from sweep_framework.data.dataset import Dataset
from sweep_framework.metrics.metric_set import MetricSet
from sweep_framework.model.loss_strategy import LossStrategy

class ModelRun:
    def __init__(
            self, 
            config: ModelConfig,
            dataset: Dataset,
            loss_strategy: Optional[LossStrategy] = None,
    ):
        self.config = config
        self.dataset = dataset
        self.loss_strategy = loss_strategy

        self.train_metrics = None
        self.val_metrics = None
        self.test_metrics = None
        self.epoch_logs = []
        self.early_stopping_triggered = False

    def train(self):
        pass

    def evaluate(self, split: str) -> MetricSet:
        return MetricSet()
    
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