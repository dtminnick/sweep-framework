# sweep-framework
A modular, extensible framework for deep learning experimentation, hyperparameter sweeps, and stakeholder-facing diagnostics. Designed for reproducibility, interpretability, and scalable model selection.

## Features
- Sweep orchestration with config grids and run groups
- Metric tracking and evaluation (macro F1, accuracy, etc.)
- Annotated reporting for stakeholder clarity
- CLI, Jupyter, and Streamlit interfaces
- Modular architecture for easy extension

## Folder Structure
- config/      # Config classes and loaders 
- model/       # Model logic and loss strategies 
- metrics/     # Evaluation and metric tracking 
- sweep/       # Sweep orchestration and registry 
- analysis/    # Reporting and stakeholder summaries 
- cli/         # Command-line interface 
- streamlit/   # Streamlit app interface 
- notebooks/   # Jupyter notebooks for exploration 
- tests/       # Unit tests 
- data/        # Sample configs or sweep results
