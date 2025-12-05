
# sweep_framework/model/focal_loss.py
"""
Implementation of Focal Loss for classification.

Focal Loss is designed to address class imbalance by down-weighting
well-classified examples and focusing training on hard examples.
"""

import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification.

    Args:
        gamma (float): Focusing parameter. Higher values increase the effect
                       of down-weighting easy examples.
        weight (Tensor, optional): Class weights for balancing.

    Example:
        >>> criterion = FocalLoss(gamma=2.0)
        >>> loss = criterion(outputs, targets)
    """

    def __init__(self, gamma: float = 2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.ce = nn.CrossEntropyLoss(weight=weight)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            input (Tensor): Model outputs (logits).
            target (Tensor): True labels.

        Returns:
            Tensor: Scalar loss value.
        """
        ce_loss = self.ce(input, target)
        pt = torch.exp(-ce_loss)
        return ((1 - pt) ** self.gamma) * ce_loss
