
import yaml
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class BaseDataset(Dataset):
    """
    Base dataset class providing reusable functionality for plan-based modeling.

    Responsibilities:
    - Load feature schema from YAML config
    - Compute normalization statistics from training data
    - Normalize values according to schema rules
    - Construct feature vectors (dynamic + static)
    - Preprocess raw plan dictionaries into tensors
    - Build PyTorch DataLoaders with custom collate function
    """

    def __init__(self, config_path="features.yaml", verbose=False):
        """
        Initialize the dataset base class.

        Args:
            config_path (str): Path to YAML config file defining feature schema.
            verbose (bool): If True, print debug logs during preprocessing.
        """
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        self.feature_schema = cfg.get("feature_schema", {})
        self.static_schema = cfg.get("static_schema", {})
        self.time_window = cfg.get("time_window", "monthly")
        self.verbose = verbose
        self.stats = {}

    def _log(self, msg):
        """
        Print a debug message if verbose mode is enabled.
        """
        if self.verbose:
            print(msg)

    def compute_stats(self, raw_data):
        """
        Compute normalization statistics (mean, std, min, max) for continuous features.

        Args:
            raw_data (list[dict]): List of plan dictionaries with "months" and "static" keys.

        Returns:
            dict: Mapping feature name → {mean, std, min, max}.
        """
        stats = {}

        # dynamic features
        for f, spec in self.feature_schema.items():
            if spec.get("type") == "continuous":
                values = []
                for plan in raw_data:
                    for period in plan["months"]:
                        if f in period:
                            values.append(period[f])
                if values:
                    stats[f] = {
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values)),
                        "min": float(np.min(values)),
                        "max": float(np.max(values)),
                    }

        # static features
        for f, spec in self.static_schema.items():
            if spec.get("type") == "continuous":
                values = [plan["static"].get(f, 0.0) for plan in raw_data]
                if values:
                    stats[f] = {
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values)),
                        "min": float(np.min(values)),
                        "max": float(np.max(values)),
                    }

        self.stats = stats
        return stats

    def normalize(self, value, feature, spec):
        """
        Normalize a single feature value according to its schema specification.

        Args:
            value (float|int|str): Raw feature value.
            feature (str): Feature name.
            spec (dict): Schema definition for the feature.

        Returns:
            float|int|torch.Tensor: Normalized value, embedding index, or one-hot vector.
        """
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
            cats = spec.get("categories", [])
            if cats:
                idx = cats.index(value) if value in cats else 0
            else:
                idx = int(value)

            if norm == "zscore":
                mu = self.stats.get(feature, {}).get("mean", 0)
                sigma = self.stats.get(feature, {}).get("std", 1)
                return (idx - mu) / sigma if sigma != 0 else 0.0
            elif norm == "minmax":
                fmin = self.stats.get(feature, {}).get("min", 0)
                fmax = self.stats.get(feature, {}).get("max", 1)
                return (idx - fmin) / (fmax - fmin) if fmax > fmin else 0.0
            else:
                return float(idx)

        if ftype == "nominal":
            cats = spec.get("categories", [])
            if cats:
                idx = cats.index(value) if value in cats else 0
                if spec.get("encoding") == "embedding":
                    return int(idx)
                elif spec.get("encoding") == "onehot":
                    vec = [1.0 if value == c else 0.0 for c in cats]
                    return torch.tensor(vec, dtype=torch.float)
            else:
                return int(value)

    def to_features(self, period_dict):
        """
        Convert a single period dictionary into a feature vector and embedding indices.

        Args:
            period_dict (dict): Dictionary of raw feature values for one time period.

        Returns:
            tuple:
                - torch.Tensor: Dense feature vector (continuous, ordinal, one-hot).
                - dict: Mapping feature → embedding index (for nominal features).
        """
        values = []
        embedding_indices = {}

        for f, spec in self.feature_schema.items():
            raw_val = period_dict.get(f, 0.0)
            enc = self.normalize(raw_val, f, spec)

            if isinstance(enc, torch.Tensor) and enc.ndim == 1 and len(enc) > 1:
                values.extend(enc.tolist())
            elif isinstance(enc, int) and spec.get("encoding") == "embedding":
                embedding_indices[f] = enc
            else:
                values.append(float(enc))

        vector = torch.tensor(values, dtype=torch.float)
        self._log(f"[Features] {period_dict.get('month','?')}: {vector.tolist()} | embeddings: {embedding_indices}")
        return vector, embedding_indices

    def preprocess_plan(self, plan):
        """
        Preprocess a full plan dictionary into tensors for model input.

        Args:
            plan (dict): Plan with keys "months", "static", and "label".

        Returns:
            tuple:
                - torch.Tensor: Dynamic sequence (time × features).
                - torch.Tensor: Static feature vector.
                - dict: Dynamic embedding indices per feature.
                - dict: Static embedding indices per feature.
                - int: Label for the plan.
        """
        periods = plan["months"]
        vectors = []
        emb_indices = {f: [] for f, spec in self.feature_schema.items() if spec.get("encoding") == "embedding"}

        # dynamic features
        for p in periods:
            vec, emb = self.to_features(p)
            vectors.append(vec)
            for f, idx in emb.items():
                emb_indices[f].append(idx)

        dynamic_seq = torch.stack(vectors)
        for f in emb_indices:
            emb_indices[f] = torch.tensor(emb_indices[f], dtype=torch.long)

        # static features
        static_vals = []
        static_embs = {}
        for f, spec in self.static_schema.items():
            raw_val = plan["static"].get(f, 0.0)
            enc = self.normalize(raw_val, f, spec)
            if isinstance(enc, int) and spec.get("encoding") == "embedding":
                static_embs[f] = torch.tensor(enc, dtype=torch.long)
            else:
                static_vals.append(float(enc))

        static_vec = torch.tensor(static_vals, dtype=torch.float)
        label = plan["label"]

        return dynamic_seq, static_vec, emb_indices, static_embs, label

    def build_loaders(self, examples, batch_size=32):
        """
        Build a PyTorch DataLoader for a list of preprocessed examples.

        Args:
            examples (list): List of tuples returned by preprocess_plan.
            batch_size (int): Number of examples per batch.

        Returns:
            DataLoader: Loader that yields batches of (sequences, static_vecs, dynamic_embs, static_embs, labels).
        """
        def collate_fn(batch):
            sequences, static_vecs, emb_dicts, static_embs, labels = zip(*batch)
            sequences = torch.stack(sequences)
            static_vecs = torch.stack(static_vecs)
            labels = torch.tensor(labels)

            batched_embs = {f: torch.stack([emb[f] for emb in emb_dicts]) for f in emb_dicts[0]}
            batched_static_embs = {f: torch.stack([emb[f] for emb in static_embs]) for f in static_embs[0]}

            return sequences, static_vecs, batched_embs, batched_static_embs, labels

        return DataLoader(examples, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
