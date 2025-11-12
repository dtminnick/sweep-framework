
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

@dataclass
class ModelConfig:
    # Architecture
    run_type: str = "LSTM"
    num_layers: int = 2
    hidden_dim: int = 128
    bidirectional: bool = True
    use_attention: bool = True
    use_layernorm: bool = True
    weightdrop_targets: Optional[str] = "weight_hh_lx"

    # Embedding
    embedding_dim: int = 100
    pretrained_embed: bool = True
    freeze_embed: bool = False
    embed_dropout: float = 0.2

    # Optimization
    learning_rate: float = 3e-4
    weight_decay: float = 5e-5
    optimizer_type: str = "Adam"
    clip_grad_norm: float = 1.0

    # Regularization
    dropout: float = 0.6
    batch_size: int = 32

    # Training
    num_epochs: int = 20
    patience: int = 5
    min_delta: float = 0.0005
    seed: int = 91210

    # Loss
    criterion: str = "CrossEntropyLoss"
    use_focal_loss: bool = False
    focal_gamma: float = 0.0

    # Metadata
    run_group: Optional[str] = None
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__
    
    def compare(self, other: "ModelConfig") -> Dict[str, Any]:
        return {k: (v, other.__dict__[k]) for k, v in self.__dict__.items() if other.__dict__.get(k) != v}
