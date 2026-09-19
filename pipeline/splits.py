"""
splits.py

Cross-validation splits for the benchmark.

Two split types are provided and both return exactly the same structure, so
the rest of the pipeline is unaware of which one is in use:

    random    stratified k-fold for classification, plain k-fold for
              regression.  This is the protocol used in the original
              submission.

    scaffold  k-fold over Bemis-Murcko scaffold groups.  Every molecule
              sharing a scaffold is placed in the same fold, so a scaffold
              seen during training never appears in the test fold.

For the multi-task dataset (ClinTox) the stratification label is the
combination of the task labels, so that the joint class distribution is
preserved rather than only the first task.
"""

from __future__ import annotations

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import KFold, StratifiedKFold

RDLogger.DisableLog("rdApp.*")


def _stratification_label(y):
    """Collapse a label array into a single label per molecule."""
    y = np.asarray(y)
    if y.ndim == 1:
        return y
    if y.shape[1] == 1:
        return y[:, 0]
    return np.array(["_".join(str(int(v)) for v in row) for row in y])


def random_folds(y, n_splits=5, seed=0, task_type="classification"):
    """Stratified (classification) or plain (regression) k-fold indices."""
    y = np.asarray(y)
    n = len(y)

    if task_type == "classification":
        labels = _stratification_label(y)
        # A class with fewer members than n_splits cannot be stratified.
        _, counts = np.unique(labels, return_counts=True)
        if counts.min() >= n_splits:
            splitter = StratifiedKFold(
                n_splits=n_splits, shuffle=True, random_state=seed
            )
            return list(splitter.split(np.zeros(n), labels))
        print(
            "[splits] a class has fewer members than the number of folds; "
            "falling back to an unstratified k-fold"
        )

    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(splitter.split(np.zeros(n)))


def murcko_scaffold(smiles, include_chirality=False):
    """Return the Bemis-Murcko scaffold of a SMILES string.

    Molecules that RDKit cannot parse, and molecules with no ring system,
    return an empty scaffold.  They are all grouped together, which keeps
    them out of both sides of a split at once.
    """
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(
            smiles=smiles, includeChirality=include_chirality
        )
    except Exception:
        return ""


def scaffold_groups(smiles_list, include_chirality=False):
    """Group molecule positions by scaffold.

    Returns a list of index arrays, one per distinct scaffold, sorted from
    the largest group to the smallest.
    """
    groups = {}
    for i, smiles in enumerate(smiles_list):
        key = murcko_scaffold(smiles, include_chirality=include_chirality)
        groups.setdefault(key, []).append(i)

    ordered = sorted(groups.values(), key=lambda g: (-len(g), g[0]))
    return [np.asarray(g, dtype=int) for g in ordered]


def scaffold_folds(smiles_list, n_splits=5, seed=0, include_chirality=False):
    """k-fold indices in which a scaffold never spans two folds.

    Scaffold groups are assigned largest first to whichever fold currently
    holds the fewest molecules, which keeps the folds close to equal in size
    while never splitting a group.  The seed shuffles groups of equal size,
    so different seeds give genuinely different scaffold partitions.
    """
    groups = scaffold_groups(smiles_list, include_chirality=include_chirality)

    rng = np.random.default_rng(seed)
    # Shuffle first, then sort by size: ties are broken differently per seed
    # while the largest-first order is preserved.
    order = rng.permutation(len(groups))
    groups = [groups[i] for i in order]
    groups.sort(key=lambda g: -len(g))

    fold_members = [[] for _ in range(n_splits)]
    fold_sizes = np.zeros(n_splits, dtype=int)

    for group in groups:
        target = int(np.argmin(fold_sizes))
        fold_members[target].append(group)
        fold_sizes[target] += len(group)

    folds = []
    for k in range(n_splits):
        test_idx = (
            np.concatenate(fold_members[k])
            if fold_members[k]
            else np.array([], dtype=int)
        )
        mask = np.ones(len(smiles_list), dtype=bool)
        mask[test_idx] = False
        train_idx = np.flatnonzero(mask)
        folds.append((train_idx, np.sort(test_idx)))

    return folds


def make_folds(split, smiles_list, y, n_splits=5, seed=0, task_type="classification"):
    """Dispatch to the requested split type."""
    if split == "random":
        return random_folds(y, n_splits=n_splits, seed=seed, task_type=task_type)
    if split == "scaffold":
        return scaffold_folds(smiles_list, n_splits=n_splits, seed=seed)
    raise ValueError(f"unknown split type: {split!r}")


def describe_folds(folds, smiles_list=None):
    """Summarise fold sizes, and scaffold overlap when SMILES are given."""
    lines = []
    for k, (train_idx, test_idx) in enumerate(folds):
        line = f"  fold {k}: train {len(train_idx)}, test {len(test_idx)}"
        if smiles_list is not None:
            train_s = {murcko_scaffold(smiles_list[i]) for i in train_idx}
            test_s = {murcko_scaffold(smiles_list[i]) for i in test_idx}
            line += f", shared scaffolds {len(train_s & test_s)}"
        lines.append(line)
    return "\n".join(lines)
