
# Overview

The **Sweep Framework** is a modular system for running machine learning experiments ("sweeps"), analyzing results, and generating unified reports.
  
It is designed to make experimentation **reproducible, extensible, and easy to analyze**.

---

## Goals

- **Reproducibility**: Every run is defined by a configuration object, ensuring experiments can be repeated exactly.
- **Extensibility**: Modular design allows new models, losses, metrics, and reporting strategies to be added easily.
- **Unified Reporting**: Results are collected into a registry, analyzed, and exported in consistent formats.

---

## Architecture

The framework is organized into modules, each with a clear responsibility:

- **Configuration (`model_config.py`)**  
  Defines hyperparameters and builds models, optimizers, and schedulers.

- **Dataset (`dataset.py`)**  
  Wraps raw examples, performs stratified splits, and builds PyTorch DataLoaders.

- **Metrics (`metric_set.py`)**  
  Tracks predictions and computes precision, recall, F1, and accuracy.

- **Loss (`loss_strategy.py`, `focal_loss.py`)**  
  Provides flexible loss functions (CrossEntropy, Focal Loss).

- **Model Run (`model_run.py`)**  
  Orchestrates training, validation, early stopping, and evaluation.

- **Sweeps (`sweep_config.py`, `sweep.py`, `sweep_registry.py`)**  
  Defines parameter grids, executes multiple runs, and stores results.

- **Analysis (`sweep_analyzer.py`)**  
  Ranks runs, compares groups, and extracts trends.

- **Reporting (`report_builder.py`)**  
  Exports summaries and insights in structured formats.

---

## Workflow

A typical workflow looks like this:

1. **Define a configuration** with `ModelConfig`.
2. **Load and split a dataset** using `Dataset`.
3. **Build loaders** with a tokenizer.
4. **Run training** with `ModelRun`.
5. **Collect results** into a `SweepRegistry`.
6. **Analyze trends** with `SweepAnalyzer`.
7. **Export reports** using `ReportBuilder`.

---

## Example (High-Level)

```python
from sweep_framework import ModelConfig, Dataset, ModelRun, SweepRegistry, SweepAnalyzer, ReportBuilder

# 1. Define configuration
config = ModelConfig(run_type="LSTM", num_layers=2, hidden_dim=128, vocab_size=30522, num_classes=2)

# 2. Load dataset
dataset = Dataset([("good movie", 1), ("bad film", 0)])
dataset.stratify_split()
dataset.build_loaders(tokenizer, batch_size=32)

# 3. Run training
run = ModelRun(config, dataset)
summary = run.run()

# 4. Collect results
registry = SweepRegistry()
registry.add_sweep("baseline", run)

# 5. Analyze
analyzer = SweepAnalyzer(registry)
df = analyzer.rank("macro_f1")

# 6. Report
report = ReportBuilder(analyzer)
print(report.export())

