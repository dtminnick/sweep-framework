
# sweep_framework/config/plan_config.py
from sweep_framework.config.base_config import BaseConfig
import torch.nn as nn

class PlanModelConfig(BaseConfig):
    """
    Model configuration for retirement plan classification.
    """

    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=1, bidirectional=False, **kwargs):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.bidirectional = bidirectional

    def build_model(self):
        return nn.Sequential(
            nn.LSTM(self.input_dim, self.hidden_dim, num_layers=self.num_layers,
                    bidirectional=self.bidirectional, batch_first=True),
            nn.Linear(self.hidden_dim * (2 if self.bidirectional else 1), self.num_classes)
        )
