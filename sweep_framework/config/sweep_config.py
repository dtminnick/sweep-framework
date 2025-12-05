
# sweep_framework/sweep/sweep_config.py
"""
SweepConfig defines the search space for hyperparameter sweeps.

It stores parameter grids and can generate all combinations for experimentation.
"""

from dataclasses import dataclass
from typing import List, Dict, Any
from itertools import product


@dataclass
class SweepConfig:
    """
    Configuration for a sweep.

    Attributes:
        name (str): Name of the sweep.
        search_space (Dict[str, List[Any]]): Parameter grid (e.g., {"hidden_dim": [128, 256]}).
        run_group (str): Group identifier for runs.
        seed (int): Random seed for reproducibility.
        notes (str): Optional notes about the sweep.
    """
    name: str
    search_space: Dict[str, List[Any]]
    run_group: str
    seed: int = 42
    notes: str = ""

    def generate_grid(self) -> List[Dict[str, Any]]:
        """
        Generate all combinations of parameters in the search space.

        Returns:
            List[Dict[str, Any]]: List of parameter dictionaries.
        """
        keys = list(self.search_space.keys())
        values = list(self.search_space.values())
        return [dict(zip(keys, combo)) for combo in product(*values)]
