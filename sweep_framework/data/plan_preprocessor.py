
import yaml
import torch

class PlanPreprocessor:
    def __init__(self, config_path="features.yaml", verbose=False, stats=None):
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        self.time_window = cfg.get("time_window", "monthly")
        self.feature_schema = cfg.get("feature_schema", {})
        self.static_schema = cfg.get("static_schema", {})
        self.verbose = verbose
        self.stats = stats or {}

    def _log(self, msg):
        if self.verbose:
            print(msg)

    def normalize(self, value, feature, spec):
        ftype = spec.get("type", "continuous")
        norm = spec.get("normalization", None)

        if ftype == "binary":
            return float(value)

        if ftype == "continuous":
            if norm == "zscore":
                mu = self.stats.get(feature, {}).get("mean", 0)
                sigma = self.stats.get(feature, {}).get("std", 1)
                return (value - mu) / sigma if sigma != 0 else 0.0
            elif norm == "minmax":
                fmin = self.stats.get(feature, {}).get("min", 0)
                fmax = self.stats.get(feature, {}).get("max", 1)
                return (value - fmin) / (fmax - fmin) if fmax > fmin else 0.0
            else:
                return float(value)

        if ftype == "ordinal":
            # treat as integer, optional normalization
            return float(value)

        if ftype == "nominal":
            # placeholder: one-hot encoding would expand dimensions
            # for now just return index value
            return float(value)

        return float(value)

    def aggregate(self, raw_periods):
        self._log(f"[Step 1] Aggregating {len(raw_periods)} periods ({self.time_window})")
        return raw_periods

    def to_features(self, period_dict):
        values = []
        for f, spec in self.feature_schema.items():
            raw_val = period_dict.get(f, 0.0)
            values.append(self.normalize(raw_val, f, spec))
        vector = torch.tensor(values, dtype=torch.float)
        self._log(f"[Step 2] Features for {period_dict.get('month','?')}: {vector.tolist()}")
        return vector

    def preprocess_plan(self, plan):
        periods = self.aggregate(plan["months"])
        dynamic_seq = torch.stack([self.to_features(p) for p in periods])
        self._log(f"[Step 3] Dynamic sequence shape: {dynamic_seq.shape}")

        static_vals = []
        for f, spec in self.static_schema.items():
            raw_val = plan["static"].get(f, 0.0)
            static_vals.append(self.normalize(raw_val, f, spec))
        static_vec = torch.tensor(static_vals, dtype=torch.float)
        self._log(f"[Step 4] Static vector: {static_vec.tolist()}")

        label = plan["label"]
        self._log(f"[Step 5] Label: {label}")

        return dynamic_seq, static_vec, label
