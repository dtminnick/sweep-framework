
# tests/test_model_config.py
import torch
from sweep_framework.config.model_config import ModelConfig

def test_model_config_builds_model():
    config = ModelConfig(
        run_type="LSTM",
        num_layers=1,
        hidden_dim=32,
        bidirectional=False,
        dropout=0.1,
        embedding_dim=50,
        vocab_size=100,
        num_classes=2,
        learning_rate=1e-3,
        optimizer_type="Adam",
        num_epochs=1,
        patience=1
    )
    model = config.build_model()
    assert isinstance(model, torch.nn.Module)

def test_optimizer_and_scheduler():
    config = ModelConfig(
        run_type="GRU",
        num_layers=1,
        hidden_dim=32,
        bidirectional=True,
        dropout=0.2,
        embedding_dim=50,
        vocab_size=100,
        num_classes=2,
        learning_rate=1e-3,
        optimizer_type="SGD",
        num_epochs=1,
        patience=1
    )
    model = config.build_model()
    optimizer = config.build_optimizer(model.parameters())
    scheduler = config.build_scheduler(optimizer)
    assert optimizer is not None
    assert scheduler is not None
