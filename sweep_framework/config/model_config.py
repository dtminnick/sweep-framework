
# sweep_framework/config/model_config.py
"""
Model configuration and builders for sequence models (LSTM/GRU).

This module defines the ModelConfig dataclass, which stores hyperparameters
and metadata for a training run. It also provides methods to build the model,
optimizer, and scheduler based on the configuration. Simple baseline LSTM and
GRU models are included for text classification tasks.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, Iterable
import torch
import torch.nn as nn
from torch.optim import Adam, SGD, AdamW
from torch.optim.lr_scheduler import StepLR


@dataclass
class ModelConfig:
    """
    Configuration object for a model run.

    Attributes:
        run_type (str): Type of RNN cell to use ("LSTM" or "GRU").
        num_layers (int): Number of recurrent layers.
        hidden_dim (int): Hidden dimension size for the RNN.
        bidirectional (bool): Whether to use bidirectional RNNs.
        use_attention (bool): Flag for attention pooling (not implemented in baseline).
        use_layernorm (bool): Whether to apply LayerNorm to RNN outputs.
        embedding_dim (int): Dimension of word embeddings.
        freeze_embed (bool): If True, embeddings are frozen during training.
        dropout (float): Dropout probability applied to RNN outputs.
        learning_rate (float): Learning rate for optimizer.
        weight_decay (float): Weight decay (L2 regularization).
        optimizer_type (str): Optimizer choice ("Adam", "SGD", "AdamW").
        clip_grad_norm (float): Gradient clipping threshold.
        num_epochs (int): Maximum number of training epochs.
        patience (int): Early stopping patience (epochs without improvement).
        seed (int): Random seed for reproducibility.
        criterion (str): Loss function name ("CrossEntropyLoss").
        use_focal_loss (bool): Whether to use focal loss.
        focal_gamma (float): Gamma parameter for focal loss.
        run_group (Optional[str]): Group identifier for sweeps.
        notes (Optional[str]): Free-form notes about the run.
        vocab_size (Optional[int]): Vocabulary size (must be set before building model).
        num_classes (int): Number of output classes.
    """

    # Architecture
    run_type: str = "LSTM"
    num_layers: int = 2
    hidden_dim: int = 128
    bidirectional: bool = True
    use_attention: bool = False
    use_layernorm: bool = False

    # Embedding
    embedding_dim: int = 100
    freeze_embed: bool = False
    dropout: float = 0.3

    # Optimization
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    optimizer_type: str = "Adam"
    clip_grad_norm: float = 1.0

    # Training
    num_epochs: int = 3
    patience: int = 2
    seed: int = 42

    # Loss
    criterion: str = "CrossEntropyLoss"
    use_focal_loss: bool = False
    focal_gamma: float = 2.0

    # Metadata
    run_group: Optional[str] = None
    notes: Optional[str] = None

    # Runtime fields
    vocab_size: Optional[int] = None
    num_classes: int = 2

    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as a dictionary."""
        return dict(self.__dict__)

    def compare(self, other: "ModelConfig") -> Dict[str, Any]:
        """Compare with another config and return differing fields."""
        return {k: (v, other.__dict__.get(k)) for k, v in self.__dict__.items() if other.__dict__.get(k) != v}

    def build_model(self) -> nn.Module:
        """
        Build the model based on run_type.

        Returns:
            nn.Module: LSTMModel or GRUModel instance.
        """
        if self.vocab_size is None:
            raise ValueError("vocab_size must be set before building the model.")
        if self.run_type.upper() == "LSTM":
            return LSTMModel(self)
        elif self.run_type.upper() == "GRU":
            return GRUModel(self)
        else:
            raise ValueError(f"Unsupported run_type: {self.run_type}")

    def build_optimizer(self, params: Iterable[torch.nn.Parameter]):
        """
        Build optimizer based on optimizer_type.

        Args:
            params (Iterable): Model parameters.

        Returns:
            torch.optim.Optimizer: Optimizer instance.
        """
        if self.optimizer_type == "Adam":
            return Adam(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer_type == "SGD":
            return SGD(params, lr=self.learning_rate, weight_decay=self.weight_decay, momentum=0.9)
        elif self.optimizer_type == "AdamW":
            return AdamW(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer_type: {self.optimizer_type}")

    def build_scheduler(self, optimizer: torch.optim.Optimizer):
        """
        Build learning rate scheduler.

        Args:
            optimizer (torch.optim.Optimizer): Optimizer to schedule.

        Returns:
            torch.optim.lr_scheduler._LRScheduler: Scheduler instance.
        """
        step_size = max(self.num_epochs // 3, 1)
        return StepLR(optimizer, step_size=step_size, gamma=0.5)


class BaseRNNModel(nn.Module):
    """
    Base class for RNN-based text classifiers.

    Architecture:
        Embedding -> RNN (LSTM/GRU) -> (LayerNorm optional) -> Dropout -> Linear classifier
    """

    def __init__(self, config: ModelConfig, cell: str = "LSTM"):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        if config.freeze_embed:
            self.embedding.weight.requires_grad = False

        rnn_cls = nn.LSTM if cell == "LSTM" else nn.GRU
        self.rnn = rnn_cls(
            input_size=config.embedding_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
            bidirectional=config.bidirectional,
        )

        out_dim = config.hidden_dim * (2 if config.bidirectional else 1)
        self.dropout = nn.Dropout(config.dropout)
        self.layernorm = nn.LayerNorm(out_dim) if config.use_layernorm else None
        self.fc = nn.Linear(out_dim, config.num_classes)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (Tensor): Input tensor of token IDs (batch_size x seq_len).

        Returns:
            Tensor: Logits for each class (batch_size x num_classes).
        """
        x = self.embedding(x)
        out, _ = self.rnn(x)
        feat = out[:, -1, :]  # last timestep
        if self.layernorm is not None:
            feat = self.layernorm(feat)
        feat = self.dropout(feat)
        return self.fc(feat)


class LSTMModel(BaseRNNModel):
    """LSTM-based text classifier."""
    def __init__(self, config: ModelConfig):
        super().__init__(config, cell="LSTM")


class GRUModel(BaseRNNModel):
    """GRU-based text classifier."""
    def __init__(self, config: ModelConfig):
        super().__init__(config, cell="GRU")
