
# ModelConfig

## Purpose
Defines hyperparameters and builder methods for models, optimizers, and schedulers.

## Attributes
- `run_type`: Model type ("LSTM" or "GRU")
- `num_layers`: Number of recurrent layers
- ...

## Usage
```python
from sweep_framework import ModelConfig
config = ModelConfig(run_type="LSTM", num_layers=2, ...)
