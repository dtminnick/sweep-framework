
from typing import Dict, List, Optional
from sweep_framework.sweep.sweep import Sweep

class SweepRegistry:
    def __init__(self):
        self.sweeps: Dict[str, Sweep] = {}
        self.groups: Dict[str, List[str]] = {}

    def add_sweep(self, sweep: Sweep):
        self.sweeps[sweep.name] = sweep
        group = sweep.run_group or "default"
        self.groups.setdefault(group, []).append(sweep.name)

    def get_sweep(self, name: str) -> Optional[Sweep]:
        return self.sweeps.get(name)

    def get_group(self, group: str) -> List[Sweep]:
        sweep_names = self.groups.get(group, [])
        return [self.sweeps[name] for name in sweep_names if name in self.sweeps]

    def list_all(self) -> List[str]:
        return list(self.sweeps.keys())

    def export_registry(self) -> Dict[str, dict]:
        return {
            name: sweep.export_results()
            for name, sweep in self.sweeps.items()
        }
