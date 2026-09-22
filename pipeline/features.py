"""
features.py

Builds the five molecular representations for a whole dataset.

Design note
-----------
Every representation is computed once per dataset and cached.  The folds
then index into the resulting arrays; nothing is recomputed per fold.

This matters for two reasons.  The practical one is cost: the benchmark
trains 5 representations x 5 folds x 3 seeds x 2 split types per dataset,
and recomputing SOAP or ACSF descriptors for each of those would dominate
the runtime.  The methodological one is that it guarantees a molecule has
exactly the same feature vector in every fold, in every seed and under both
split types, so any difference in the results comes from the split or the
training and not from the featurisation.

A consequence is that the padded sizes (the largest molecule, the largest
bag, the largest number of views) are taken over the whole dataset rather
than over each training fold.  These sizes carry no label information, and
fixing them dataset-wide also removes the truncation of large test
molecules that a per-fold size would otherwise cause.
"""

from __future__ import annotations

import os
import pickle
import time

import numpy as np

from featurization.acsf import create_acsf
from featurization.bag_of_bonds import create_bob
from featurization.coulomb_matrix import create_coulomb_matrix
from featurization.soap import create_soap
from featurization.weightedviews import load_data, speciesmap

# The four representations that produce one flat vector per molecule and are
# fed to an ordinary dense network.
DENSE_REPRESENTATIONS = ["cm", "bob", "acsf", "soap"]

# Weighted Views produces a set of views per molecule and uses its own
# architecture, so it is handled separately throughout the pipeline.
ALL_REPRESENTATIONS = DENSE_REPRESENTATIONS + ["views"]

DISPLAY_NAMES = {
    "cm": "Coulomb Matrix",
    "bob": "Bag of Bonds",
    "acsf": "ACSF",
    "soap": "SOAP",
    "views": "Weighted Views",
}


def build_cm(mollist):
    """Sorted Coulomb Matrix, flattened to one vector per molecule."""
    matrices = create_coulomb_matrix(mollist, max_atoms=None, sort=True)
    return matrices.reshape(matrices.shape[0], -1)


def build_bob(mollist):
    """Bag of Bonds, with bag sizes taken over the whole dataset."""
    features, _bag_sizes = create_bob(mollist, bag_sizes=None)
    return features


def dataset_species(mollist):
    """The chemical elements that actually occur in a dataset.

    ACSF and SOAP both size their descriptors from the list of species they
    are told to expect, and both grow quickly with it: SOAP roughly with the
    square of the number of species, ACSF likewise through its angular
    terms.  Passing a fixed list of every element the benchmark might ever
    meet therefore produces a descriptor that is mostly structural zeros,
    and one so wide that matching a parameter budget forces an unusably
    narrow network.  Taking the species from the dataset keeps the
    descriptor to the size the chemistry actually requires.
    """
    return sorted({atom[0] for molecule in mollist for atom in molecule})


def build_acsf(mollist):
    """ACSF descriptors, zero padded to the largest molecule and flattened.

    ACSF gives one vector per atom.  The atomic vectors are stacked, padded
    with zero rows up to the largest molecule in the dataset, and flattened
    into a single vector, which is the pooling the paper describes.
    """
    features, _max_atoms = create_acsf(
        mollist, max_atoms=None, species=dataset_species(mollist)
    )
    return features.reshape(features.shape[0], -1)


def build_soap(mollist):
    """SOAP power spectra averaged over the atoms of each molecule.

    The averaging is performed inside dscribe through ``average="outer"``,
    so this already returns one fixed length vector per molecule.
    """
    return create_soap(mollist, species=dataset_species(mollist))


def build_views(mollist):
    """Weighted Views weights and view matrices for a whole dataset.

    Returns ``(weights, views, n_atoms, n_views)``.  Note that the original
    notebooks called ``load_data`` for the test set with ``setNviews`` set
    to the number of atoms rather than the number of views, which is a
    different quantity.  Computing the whole dataset in one call removes
    that mistake, because the padded sizes are then consistent by
    construction.
    """
    weights, views, n_atoms, n_views = load_data(
        mollist,
        setNatoms=None,
        setNviews=None,
        carbonbased=False,
        splitx=False,
        verbose=0,
    )
    return weights, views, n_atoms, n_views


BUILDERS = {
    "cm": build_cm,
    "bob": build_bob,
    "acsf": build_acsf,
    "soap": build_soap,
}


def build_representation(name, mollist):
    """Build one representation for a whole dataset, with timing."""
    start = time.perf_counter()

    if name == "views":
        weights, views, n_atoms, n_views = build_views(mollist)
        payload = {
            "weights": weights,
            "views": views,
            "n_atoms": int(n_atoms),
            "n_views": int(n_views),
        }
        shape = views.shape
    elif name in BUILDERS:
        features = BUILDERS[name](mollist)
        payload = {"X": features}
        shape = features.shape
    else:
        raise ValueError(f"unknown representation: {name!r}")

    payload["build_seconds"] = time.perf_counter() - start
    payload["shape"] = tuple(int(s) for s in shape)

    # A descriptor should never contain an infinity or a NaN. If one appears
    # it usually means two atoms ended up at the same coordinates, making an
    # interatomic distance zero, and it is far easier to diagnose here than
    # from a model that silently predicts nonsense.
    for key in ("X", "views", "weights"):
        if key in payload:
            bad = int((~np.isfinite(payload[key])).sum())
            if bad:
                raise ValueError(
                    f"{name} produced {bad} non-finite values in '{key}'; "
                    "check the geometries for overlapping atoms"
                )

    return payload


# Bumped whenever a change to this module makes previously cached
# descriptors invalid, so that a stale cache is never silently reused.
CACHE_VERSION = 2


def load_or_build_representation(dataset, name, mollist, cache_dir="cache"):
    """Return a cached representation, building and caching it if needed."""
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{dataset}_{name}_v{CACHE_VERSION}.pkl")

    if os.path.exists(path):
        with open(path, "rb") as handle:
            payload = pickle.load(handle)
        print(f"[features] loaded {name} for {dataset} from cache, shape {payload['shape']}")
        return payload

    print(f"[features] building {name} for {dataset} ...")
    payload = build_representation(name, mollist)
    with open(path, "wb") as handle:
        pickle.dump(payload, handle)
    print(
        f"[features] {name} for {dataset}: shape {payload['shape']}, "
        f"{payload['build_seconds']:.1f} s"
    )
    return payload
