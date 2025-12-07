import yaml
import torch

class PlanPreprocessor:
    def __init__(self, config_path="features.yaml", verbose=False):
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        self.time_window = cfg.get("time_window", "monthly")
        self.feature_schema = cfg.get("feature_schema", [])
        self.static_schema = cfg.get("static_schema", [])
        self.verbose = verbose

    def _log(self, msg):
        if self.verbose:
            print(msg)

    def aggregate(self, raw_periods):
        # For now, assume raw_periods already match time_window
        self._log(f"[Step 1] Aggregating {len(raw_periods)} periods ({self.time_window})")
        return raw_periods

    def to_features(self, period_dict):
        vector = torch.tensor([period_dict.get(f, 0.0) for f in self.feature_schema], dtype=torch.float)
        self._log(f"[Step 2] Features for {period_dict['month']}: {vector.tolist()}")
        return vector

    def preprocess_plan(self, plan):
        periods = self.aggregate(plan["months"])
        dynamic_seq = torch.stack([self.to_features(p) for p in periods])
        self._log(f"[Step 3] Dynamic sequence shape: {dynamic_seq.shape}")

        static_vec = torch.tensor([plan["static"].get(f, 0.0) for f in self.static_schema], dtype=torch.float)
        self._log(f"[Step 4] Static vector: {static_vec.tolist()}")

        label = plan["label"]
        self._log(f"[Step 5] Label: {label}")

        return dynamic_seq, static_vec, label


