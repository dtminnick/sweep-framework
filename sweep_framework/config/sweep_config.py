
# sweep_framework/config/sweep_config.py

from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class SweepConfig:
    name: str
    search_space: Dict[str, List[Any]]
    run_group: str
    seed: int = 42
    notes: str = ""

    def generate_grid(self) -> List[Dict[str, Any]]:
        from itertools import product
        keys = list(self.search_space.keys())
        values = list(self.search_space.values())
        return [dict(zip(keys, combo)) for combo in product(*values)]
