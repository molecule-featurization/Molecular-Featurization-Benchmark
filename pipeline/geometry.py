"""
geometry.py

Three-dimensional geometry generation for the benchmark.

SMILES strings are embedded with ETKDGv3 and relaxed with UFF, exactly as
described in the paper.  Two things differ from the original notebooks:

1.  The embedding is seeded, so the geometries are reproducible.  The
    conformer seed is deliberately kept separate from the seeds used for
    cross-validation, so that every run of the benchmark sees the same
    molecular geometries and only the data split and the weight
    initialisation change between seeds.

2.  Every molecule that is dropped is recorded with a reason, so that the
    number of excluded molecules and the final sample size can be reported.

The geometries are cached on disk.  Conformer generation is by far the most
expensive preprocessing step, and caching means it happens once per dataset
rather than once per representation, split and fold.
"""

from __future__ import annotations

import os
import pickle
from collections import Counter

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

# RDKit prints a great deal of noise for molecules it cannot parse.
RDLogger.DisableLog("rdApp.*")


DEFAULT_CONFORMER_SEED = 0xF00D


def _has_duplicate_coordinates(mol, conf, decimals=6):
    """Return True if two atoms share the same position.

    A collapsed geometry of this kind makes the interatomic distance zero,
    which produces infinities in the Coulomb Matrix and Bag of Bonds.  The
    original notebooks applied the same test.
    """
    seen = set()
    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        key = (round(pos.x, decimals), round(pos.y, decimals), round(pos.z, decimals))
        if key in seen:
            return True
        seen.add(key)
    return False


def embed_molecule(smiles, seed=DEFAULT_CONFORMER_SEED, max_attempts=10):
    """Embed one SMILES string in three dimensions.

    Returns
    -------
    (molecule, reason)
        ``molecule`` is a list of ``(element_symbol, xyz)`` tuples, or None
        if the molecule could not be embedded.  ``reason`` is None on
        success and otherwise a short string naming the failure.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, "invalid_smiles"

    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    # The attribute name has changed across RDKit versions, so set whichever
    # one this build exposes rather than assuming.
    for attribute in ("maxIterations", "maxAttempts"):
        if hasattr(params, attribute):
            setattr(params, attribute, int(max_attempts))
            break

    if AllChem.EmbedMolecule(mol, params) != 0:
        return None, "embedding_failed"

    try:
        AllChem.UFFOptimizeMolecule(mol)
    except Exception:
        return None, "uff_failed"

    conf = mol.GetConformer()

    if _has_duplicate_coordinates(mol, conf):
        return None, "duplicate_coordinates"

    molecule = []
    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        molecule.append(
            (
                atom.GetSymbol(),
                np.array([pos.x, pos.y, pos.z], dtype=np.float32),
            )
        )

    return molecule, None


def build_geometries(smiles_list, y, seed=DEFAULT_CONFORMER_SEED):
    """Embed a whole dataset, keeping labels aligned with the geometries.

    Returns
    -------
    mollist : list
        One entry per successfully embedded molecule.
    y_kept : np.ndarray
        The labels of those molecules, in the same order.
    kept_index : np.ndarray
        Positions in the original ``smiles_list`` that survived.
    failures : dict
        Reason -> count, for reporting how many molecules were excluded.
    """
    y = np.asarray(y)

    mollist = []
    kept_index = []
    failures = Counter()

    for i, smiles in enumerate(smiles_list):
        molecule, reason = embed_molecule(smiles, seed=seed)
        if molecule is None:
            failures[reason] += 1
            continue
        mollist.append(molecule)
        kept_index.append(i)

    kept_index = np.asarray(kept_index, dtype=int)
    y_kept = y[kept_index]

    return mollist, y_kept, kept_index, dict(failures)


def load_or_build_geometries(
    name,
    smiles_list,
    y,
    seed=DEFAULT_CONFORMER_SEED,
    cache_dir="cache",
):
    """Return cached geometries for a dataset, building them if needed."""
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{name}_geometry_seed{seed}.pkl")

    if os.path.exists(path):
        with open(path, "rb") as handle:
            payload = pickle.load(handle)
        print(f"[geometry] loaded {len(payload['mollist'])} molecules from {path}")
        return (
            payload["mollist"],
            payload["y"],
            payload["kept_index"],
            payload["failures"],
        )

    print(f"[geometry] embedding {len(smiles_list)} molecules for {name} ...")
    mollist, y_kept, kept_index, failures = build_geometries(smiles_list, y, seed=seed)

    payload = {
        "mollist": mollist,
        "y": y_kept,
        "kept_index": kept_index,
        "failures": failures,
        "seed": seed,
    }
    with open(path, "wb") as handle:
        pickle.dump(payload, handle)

    n_failed = sum(failures.values())
    print(
        f"[geometry] {name}: kept {len(mollist)} of {len(smiles_list)} molecules "
        f"({n_failed} excluded: {failures})"
    )
    return mollist, y_kept, kept_index, failures
