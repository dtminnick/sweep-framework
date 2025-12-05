
# sweep_framework/model/loss_strategy.py
"""
Loss strategy builder for model runs.

This module defines LossStrategy, which selects and constructs the appropriate
loss function (CrossEntropy or Focal Loss) based on ModelConfig.
"""

import torch
import torch.nn as nn
from typing import Optional
from sweep_framework.config.model_config import ModelConfig
from sweep_framework.model.focal_loss import FocalLoss


class LossStrategy:
    """
    Strategy for selecting and computing loss.

    Attributes:
        loss_type (str): Name of loss function ("CrossEntropyLoss").
        use_focal (bool): Whether to use focal loss.
        focal_gamma (float): Gamma parameter for focal loss.
        class_weights (Optional[Tensor]): Class weights for balancing.
    """

    def __init__(
        self,
        loss_type: str = "CrossEntropyLoss",
        use_focal: bool = False,
        focal_gamma: float = 2.0,
        class_weights: Optional[torch.Tensor] = None,
    ):
        self.loss_type = loss_type
        self.use_focal = use_focal
        self.focal_gamma = focal_gamma
        self.class_weights = class_weights

    def build(self) -> nn.Module:
        """
        Build the loss function.

        Returns:
            nn.Module: Loss function instance.
        """
        if self.use_focal:
            return FocalLoss(gamma=self.focal_gamma, weight=self.class_weights)
        if self.loss_type == "CrossEntropyLoss":
            return nn.CrossEntropyLoss(weight=self.class_weights)
        raise ValueError(f"Unsupported loss type: {self.loss_type}")

    def compute(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute loss for a batch.

        Args:
            outputs (Tensor): Model outputs (logits).
            targets (Tensor): True labels.

        Returns:
            Tensor: Scalar loss value.
        """
        return self.build()(outputs, targets)

    def describe(self) -> dict:
        """
        Return metadata about the loss strategy.

        Returns:
            dict: Description of loss type and parameters.
        """
        return {
            "loss_type": self.loss_type,
            "use_focal": self.use_focal,
            "focal_gamma": self.focal_gamma,
            "has_class_weights": self.class_weights is not None,
        }

    @classmethod
    def from_config(cls, config: ModelConfig) -> "LossStrategy":
        """
        Construct LossStrategy from a ModelConfig.

        Args:
            config (ModelConfig): Configuration object.

        Returns:
            LossStrategy: Initialized strategy.
        """
        return cls(
            loss_type=config.criterion,
            use_focal=config.use_focal_loss,
            focal_gamma=config.focal_gamma,
            class_weights=None,  # weights can be injected later
        )
