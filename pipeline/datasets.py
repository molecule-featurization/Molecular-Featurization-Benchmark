"""
datasets.py

Loads the five MoleculeNet benchmark datasets as plain SMILES and labels.

DeepChem is used only as the data source, with ``splitter=None`` so that the
full dataset comes back unsplit and this pipeline controls the splitting.
If DeepChem is not installed, the loader falls back to reading the same
datasets from a local CSV in ``data/``, so the benchmark can be run without
it.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

DATASETS = {
    "esol": {
        "loader": "load_delaney",
        "task_type": "regression",
        "metric": "rmse",
        "smiles_column": "smiles",
    },
    "freesolv": {
        "loader": "load_sampl",
        "task_type": "regression",
        "metric": "rmse",
        "smiles_column": "smiles",
    },
    "bbbp": {
        "loader": "load_bbbp",
        "task_type": "classification",
        "metric": "roc_auc",
        "smiles_column": "smiles",
    },
    "bace": {
        "loader": "load_bace_classification",
        "task_type": "classification",
        "metric": "roc_auc",
        "smiles_column": "mol",
    },
    "clintox": {
        "loader": "load_clintox",
        "task_type": "classification",
        "metric": "roc_auc",
        "smiles_column": "smiles",
    },
}


def _load_from_deepchem(name):
    import deepchem as dc

    loader = getattr(dc.molnet, DATASETS[name]["loader"])
    _tasks, datasets, _transformers = loader(splitter=None, featurizer="Raw")
    dataset = datasets[0]

    smiles = list(dataset.ids)
    y = np.asarray(dataset.y, dtype=np.float32)
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    return smiles, y


def _load_from_csv(name, data_dir="data"):
    path = os.path.join(data_dir, f"{name}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Either install deepchem, or place a CSV at "
            f"{path} with a SMILES column and one column per task."
        )

    frame = pd.read_csv(path)
    smiles_column = DATASETS[name]["smiles_column"]
    if smiles_column not in frame.columns:
        candidates = [c for c in frame.columns if c.lower() in ("smiles", "mol")]
        if not candidates:
            raise ValueError(f"no SMILES column found in {path}")
        smiles_column = candidates[0]

    smiles = frame[smiles_column].astype(str).tolist()
    label_columns = [c for c in frame.columns if c != smiles_column]
    y = frame[label_columns].to_numpy(dtype=np.float32)
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    return smiles, y


def load_dataset(name, data_dir="data"):
    """Return ``(smiles, y, info)`` for one benchmark dataset."""
    if name not in DATASETS:
        raise ValueError(
            f"unknown dataset {name!r}; expected one of {sorted(DATASETS)}"
        )

    try:
        smiles, y = _load_from_deepchem(name)
        source = "deepchem"
    except ImportError:
        smiles, y = _load_from_csv(name, data_dir=data_dir)
        source = "csv"

    info = dict(DATASETS[name])
    info["name"] = name
    info["source"] = source
    info["n_molecules"] = len(smiles)
    info["n_tasks"] = int(y.shape[1])

    return smiles, y, info
