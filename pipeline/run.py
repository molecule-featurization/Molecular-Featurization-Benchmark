"""
run.py

Main entry point for the benchmark.

One invocation evaluates every requested representation on one dataset
under one split type, across every fold of every seed, and appends one row
per fold to a CSV.  Nothing is aggregated here; ``report.py`` turns the
per-fold rows into the tables that go in the paper.

Typical use:

    python -m pipeline.run --dataset bbbp --split random
    python -m pipeline.run --dataset bbbp --split scaffold
    python -m pipeline.run --all

The per-fold rows are what make the paired comparisons possible: the same
dataset, seed and fold index appear under both split types and under every
representation, so differences can be taken fold by fold rather than only
between means.
"""

from __future__ import annotations

import argparse
import logging
import os
import time
import warnings

import numpy as np
import pandas as pd

# A fresh model is built for every fold, which is intentional, so Keras'
# retracing notice is expected here and only obscures the results.
logging.getLogger("tensorflow").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*retracing.*")
warnings.filterwarnings("ignore", message=".*structure of `inputs`.*")
from sklearn.metrics import roc_auc_score

from pipeline.datasets import DATASETS, load_dataset
from pipeline.features import (
    ALL_REPRESENTATIONS,
    DENSE_REPRESENTATIONS,
    load_or_build_representation,
)
from pipeline.geometry import DEFAULT_CONFORMER_SEED, load_or_build_geometries
from pipeline.models import (
    build_dense,
    build_weighted_views,
    count_trainable_parameters,
    set_seeds,
)
from pipeline.splits import make_folds, murcko_scaffold

RESULT_COLUMNS = [
    "dataset", "representation", "split", "seed", "fold",
    "metric", "value", "train_value",
    "trainable_parameters", "hidden_width", "input_dim", "standardized",
    "n_low_variance_features",
    "n_train", "n_test", "epochs", "batch_size",
    "fit_seconds", "predict_seconds",
]


def standardize_fold(x_train, x_test, clip=10.0, min_std=1e-8):
    """Standardise features using training-fold statistics only.

    Two guards matter here, both because of zero padding.  ACSF pads every
    molecule out to the largest one in the dataset, so a great many columns
    are zero for almost every molecule in a given training fold.  Such a
    column has a standard deviation close to zero, and dividing by it turns
    any test molecule that happens to use that position into a value of
    order 1e5, which three ReLU layers then drive to infinity.

    So a column whose training standard deviation is negligible is left
    unscaled rather than divided by, and the standardised values are capped
    at ``clip`` standard deviations.  Genuine standardised features are
    almost never beyond about five, so the cap only ever bites on the
    pathological out-of-range cases it is there to catch.
    """
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)

    weak = std < min_std
    mean = np.where(weak, 0.0, mean)
    std = np.where(weak, 1.0, std)

    z_train = np.clip((x_train - mean) / std, -clip, clip)
    z_test = np.clip((x_test - mean) / std, -clip, clip)

    return z_train.astype(np.float32), z_test.astype(np.float32), int(weak.sum())


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mean_roc_auc(y_true, y_pred):
    """ROC-AUC, averaged over tasks for a multi-task dataset.

    A task whose test fold contains only one class has no defined ROC-AUC
    and is skipped rather than counted as zero.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)

    scores = []
    for task in range(y_true.shape[1]):
        column = y_true[:, task]
        if len(np.unique(column)) < 2:
            continue
        scores.append(roc_auc_score(column, y_pred[:, task]))

    return float(np.mean(scores)) if scores else float("nan")


def score(y_true, y_pred, task_type):
    return mean_roc_auc(y_true, y_pred) if task_type == "classification" else rmse(y_true, y_pred)


def run_fold(representation, payload, y, train_idx, test_idx, task_type, seed,
             target_params=None, epochs=150, batch_size=32, l2=0.01,
             aggregation="learned", standardize=True):
    """Train one representation on one fold and return a result row."""
    set_seeds(seed)

    output_dim = y.shape[1]
    n_low_variance = 0

    if representation == "views":
        weights = payload["weights"]
        views = payload["views"]
        x_train = [weights[train_idx], views[train_idx]]
        x_test = [weights[test_idx], views[test_idx]]
        input_dim = int(views.shape[2])

        model = build_weighted_views(
            view_dim=input_dim,
            n_views=int(views.shape[1]),
            output_dim=output_dim,
            task_type=task_type,
            l2=l2,
            aggregation=aggregation,
        )
        width = None
    else:
        features = payload["X"]
        x_train = features[train_idx]
        x_test = features[test_idx]
        input_dim = int(features.shape[1])

        if standardize:
            # The four flat representations live on very different scales:
            # Coulomb Matrix entries are Z_i Z_j / R and run to the
            # hundreds, while SOAP power spectrum entries are of order one.
            # Without scaling, a representation is judged partly on its
            # units rather than on the structure it encodes.  The statistics
            # come from the training fold only, so nothing about the test
            # fold enters the model.
            #
            # Weighted Views is left alone: its inputs are atomic
            # coordinates and species indicators, which are already on a
            # common scale, and centring them would destroy the meaning of
            # the zero padding used for absent atoms.
            x_train, x_test, n_low_variance = standardize_fold(x_train, x_test)

        model, width = build_dense(
            input_dim=input_dim,
            output_dim=output_dim,
            task_type=task_type,
            hidden_layers=3,
            target_params=target_params,
            l2=l2,
        )

    y_train = y[train_idx]
    y_test = y[test_idx]

    start = time.perf_counter()
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    fit_seconds = time.perf_counter() - start

    start = time.perf_counter()
    pred_test = model.predict(x_test, verbose=0)
    predict_seconds = time.perf_counter() - start
    pred_train = model.predict(x_train, verbose=0)

    row = {
        "representation": representation,
        "seed": seed,
        "metric": "roc_auc" if task_type == "classification" else "rmse",
        "value": score(y_test, pred_test, task_type),
        "train_value": score(y_train, pred_train, task_type),
        "trainable_parameters": count_trainable_parameters(model),
        "hidden_width": width,
        "input_dim": input_dim,
        "standardized": bool(standardize) and representation != "views",
        "n_low_variance_features": n_low_variance,
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "epochs": epochs,
        "batch_size": batch_size,
        "fit_seconds": round(fit_seconds, 2),
        "predict_seconds": round(predict_seconds, 3),
    }

    keras_backend_clear()
    return row


def keras_backend_clear():
    """Release the graph between folds so memory does not accumulate."""
    from tensorflow import keras

    keras.backend.clear_session()


def run_dataset(dataset, split, representations=None, seeds=(0, 1, 2), n_splits=5,
                epochs=150, batch_size=32, l2=0.01, standardize=True,
                conformer_seed=DEFAULT_CONFORMER_SEED,
                cache_dir="cache", results_dir="results"):
    """Run every representation on one dataset under one split type."""
    representations = list(representations or ALL_REPRESENTATIONS)

    smiles_all, y_all, info = load_dataset(dataset)
    task_type = info["task_type"]

    mollist, y, kept_index, failures = load_or_build_geometries(
        dataset, smiles_all, y_all, seed=conformer_seed, cache_dir=cache_dir
    )
    smiles = [smiles_all[i] for i in kept_index]

    payloads = {
        name: load_or_build_representation(dataset, name, mollist, cache_dir=cache_dir)
        for name in representations
    }

    rows = []
    for seed in seeds:
        folds = make_folds(
            split, smiles, y, n_splits=n_splits, seed=seed, task_type=task_type
        )

        for fold_index, (train_idx, test_idx) in enumerate(folds):
            # The Weighted Views model defines the parameter budget, so it is
            # built first and its count is the target for the other four.
            target_params = None
            if "views" in representations:
                probe = build_weighted_views(
                    view_dim=int(payloads["views"]["views"].shape[2]),
                    n_views=int(payloads["views"]["views"].shape[1]),
                    output_dim=int(y.shape[1]),
                    task_type=task_type,
                    l2=l2,
                )
                target_params = count_trainable_parameters(probe)
                keras_backend_clear()

            for name in representations:
                if name in DENSE_REPRESENTATIONS and target_params is None:
                    raise ValueError(
                        "the dense representations are sized against the "
                        "Weighted Views parameter count, so 'views' must be "
                        "among the representations being run"
                    )

                row = run_fold(
                    representation=name,
                    payload=payloads[name],
                    y=y,
                    train_idx=train_idx,
                    test_idx=test_idx,
                    task_type=task_type,
                    seed=seed,
                    target_params=target_params,
                    epochs=epochs,
                    batch_size=batch_size,
                    l2=l2,
                    standardize=standardize,
                )
                row.update({"dataset": dataset, "split": split, "fold": fold_index})
                rows.append(row)

                print(
                    f"  {dataset:9s} {split:8s} seed {seed} fold {fold_index} "
                    f"{name:6s} {row['metric']} = {row['value']:.4f} "
                    f"({row['trainable_parameters']:,} params, {row['fit_seconds']:.0f}s)"
                )

    frame = pd.DataFrame(rows)[RESULT_COLUMNS]

    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, "per_fold_results.csv")
    header = not os.path.exists(path)
    frame.to_csv(path, mode="a", header=header, index=False)
    print(f"[run] appended {len(frame)} rows to {path}")

    write_dataset_summary(
        dataset, info, smiles_all, smiles, failures, payloads, results_dir
    )

    return frame


def write_dataset_summary(dataset, info, smiles_all, smiles_kept, failures,
                          payloads, results_dir):
    """Record the counts and descriptor sizes the reviewers asked about."""
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, "dataset_summary.csv")

    record = {
        "dataset": dataset,
        "task_type": info["task_type"],
        "n_tasks": info["n_tasks"],
        "n_molecules_raw": len(smiles_all),
        "n_molecules_used": len(smiles_kept),
        "n_excluded": len(smiles_all) - len(smiles_kept),
        "exclusion_reasons": "; ".join(f"{k}={v}" for k, v in sorted(failures.items())) or "none",
        "n_scaffolds": len({murcko_scaffold(s) for s in smiles_kept}),
    }
    for name, payload in payloads.items():
        record[f"{name}_dim"] = payload["shape"][-1]
        record[f"{name}_build_seconds"] = round(payload["build_seconds"], 2)

    frame = pd.DataFrame([record])
    if os.path.exists(path):
        existing = pd.read_csv(path)
        existing = existing[existing["dataset"] != dataset]
        frame = pd.concat([existing, frame], ignore_index=True)
    frame.to_csv(path, index=False)
    print(f"[run] wrote dataset summary to {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASETS), help="dataset to run")
    parser.add_argument("--split", choices=["random", "scaffold"], default="random")
    parser.add_argument("--all", action="store_true",
                        help="run every dataset under both split types")
    parser.add_argument("--representations", nargs="+", default=ALL_REPRESENTATIONS,
                        choices=ALL_REPRESENTATIONS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--l2", type=float, default=0.01)
    parser.add_argument("--no-standardize", action="store_true",
                        help="skip per-fold feature standardisation")
    parser.add_argument("--conformer-seed", type=int, default=DEFAULT_CONFORMER_SEED)
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--results-dir", default="results")

    args = parser.parse_args()

    if not args.all and not args.dataset:
        parser.error("pass --dataset, or --all to run everything")

    jobs = (
        [(d, s) for d in sorted(DATASETS) for s in ("random", "scaffold")]
        if args.all
        else [(args.dataset, args.split)]
    )

    for dataset, split in jobs:
        print(f"\n=== {dataset} | {split} split ===")
        run_dataset(
            dataset=dataset,
            split=split,
            representations=args.representations,
            seeds=tuple(args.seeds),
            n_splits=args.folds,
            epochs=args.epochs,
            batch_size=args.batch_size,
            l2=args.l2,
            standardize=not args.no_standardize,
            conformer_seed=args.conformer_seed,
            cache_dir=args.cache_dir,
            results_dir=args.results_dir,
        )


if __name__ == "__main__":
    main()
