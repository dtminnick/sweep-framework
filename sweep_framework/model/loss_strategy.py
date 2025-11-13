
import torch
import torch.nn as nn
from typing import Optional
from sweep_framework.config.model_config import ModelConfig
from sweep_framework.model.focal_loss import FocalLoss

class LossStrategy:
    def __init__(
        self,
        loss_type: str = "CrossEntropyLoss",
        use_focal: bool = False,
        focal_gamma: float = 0.0,
        class_weights: Optional[torch.Tensor] = None,
    ):
        self.loss_type = loss_type
        self.use_focal = use_focal
        self.focal_gamma = focal_gamma
        self.class_weights = class_weights

    def build(self) -> nn.Module:
        if self.use_focal:
            return FocalLoss(gamma=self.focal_gamma, weight=self.class_weights)
        if self.loss_type == "CrossEntropyLoss":
            return nn.CrossEntropyLoss(weight=self.class_weights)
        raise ValueError(f"Unsupported loss type: {self.loss_type}")

    def describe(self) -> dict:
        return {
            "loss_type": self.loss_type,
            "use_focal": self.use_focal,
            "focal_gamma": self.focal_gamma,
            "has_class_weights": self.class_weights is not None,
        }

    @classmethod
    def from_config(cls, config: ModelConfig) -> "LossStrategy":
        return cls(
            loss_type=config.criterion,
            use_focal=config.use_focal_loss,
            focal_gamma=config.focal_gamma,
            class_weights=None,  # You can inject weights later if needed
        )
    