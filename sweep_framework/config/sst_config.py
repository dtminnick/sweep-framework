
# sweep_framework/config/sst_config.py
from sweep_framework.config.base_config import BaseConfig
import torch.nn as nn

class SSTModelConfig(BaseConfig):
    """
    Model configuration for text classification (SST).
    """

    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, **kwargs):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        # other hyperparameters...

    def build_model(self):
        return nn.Sequential(
            nn.Embedding(self.vocab_size, self.embedding_dim),
            nn.LSTM(self.embedding_dim, self.hidden_dim, batch_first=True),
            nn.Linear(self.hidden_dim, self.num_classes)
        )
