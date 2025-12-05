
# sweep_framework/sweep/sweep_registry.py
"""
SweepRegistry manages multiple sweeps and groups.

Responsibilities:
- Store sweeps by name.
- Organize sweeps into groups.
- Retrieve sweeps or groups.
- Export registry for reporting.
"""

from typing import Dict, List, Optional
from sweep_framework.sweep.sweep import Sweep


class SweepRegistry:
    """
    Registry for managing sweeps.

    Attributes:
        sweeps (Dict[str, Sweep]): Mapping of sweep names to Sweep objects.
        groups (Dict[str, List[str]]): Mapping of group names to sweep names.
    """

    def __init__(self):
        self.sweeps: Dict[str, Sweep] = {}
        self.groups: Dict[str, List[str]] = {}

    def add_sweep(self, sweep: Sweep):
        """
        Add a sweep to the registry.

        Args:
            sweep (Sweep): Sweep to add.
        """
        self.sweeps[sweep.name] = sweep
        group = sweep.run_group or "default"
        self.groups.setdefault(group, []).append(sweep.name)

    def get_sweep(self, name: str) -> Optional[Sweep]:
        """
        Retrieve a sweep by name.

        Args:
            name (str): Sweep name.

        Returns:
            Sweep or None.
        """
        return self.sweeps.get(name)

    def get_group(self, group: str) -> List[Sweep]:
        """
        Retrieve all sweeps in a group.

        Args:
            group (str): Group name.

        Returns:
            List[Sweep]: Sweeps in the group.
        """
        sweep_names = self.groups.get(group, [])
        return [self.sweeps[name] for name in sweep_names if name in self.sweeps]

    def list_all(self) -> List[str]:
        """
        List all sweep names.

        Returns:
            List[str]: Names of sweeps.
        """
        return list(self.sweeps.keys())

    def export_registry(self) -> Dict[str, dict]:
        """
        Export all sweeps in the registry.

        Returns:
            Dict[str, dict]: Sweep results keyed by name.
        """
        return {name: sweep.export_results() for name, sweep in self.sweeps.items()}
