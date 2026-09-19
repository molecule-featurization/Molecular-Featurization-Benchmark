"""
report.py

Turns the per-fold results into the tables the paper needs.

Four outputs are produced:

    main_results        mean and standard deviation per representation,
                        dataset and split type

    paired_deltas       per-fold differences between representations on
                        matching folds, with the mean difference and a
                        paired t-test

    split_effect        how much each representation moves when only the
                        split changes, compared with how far apart the
                        representations are within one split

    architecture        input dimension, hidden width and trainable
                        parameter count per representation and dataset

The paired and split-effect tables exist because differences taken on
matching folds are far less noisy than differences between independently
computed means, and because the claim that the protocol matters more than
the representation needs the two quantities side by side.
"""

from __future__ import annotations

import argparse
import itertools
import os

import numpy as np
import pandas as pd

from pipeline.features import DISPLAY_NAMES

FOLD_KEYS = ["dataset", "split", "seed", "fold"]


def load_results(path="results/per_fold_results.csv"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found; run pipeline.run first")
    return pd.read_csv(path)


def main_results(frame):
    """Mean and standard deviation over all folds and seeds."""
    grouped = (
        frame.groupby(["dataset", "split", "representation", "metric"])["value"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    grouped["reported"] = grouped.apply(
        lambda r: f"{r['mean']:.3f} ± {r['std']:.3f}", axis=1
    )
    grouped["representation"] = grouped["representation"].map(DISPLAY_NAMES)
    return grouped


def results_table(frame, split):
    """A wide table in the layout used in the paper."""
    subset = frame[frame["split"] == split]
    summary = main_results(subset)
    return summary.pivot_table(
        index="dataset", columns="representation", values="reported", aggfunc="first"
    )


def paired_deltas(frame):
    """Differences between representations on matching folds.

    Every pair of representations is compared fold by fold, within the same
    dataset, split, seed and fold index.  For RMSE a negative difference
    means the first representation has the lower error; for ROC-AUC a
    positive difference means the first has the higher score.
    """
    from scipy import stats

    rows = []
    for (dataset, split), block in frame.groupby(["dataset", "split"]):
        wide = block.pivot_table(
            index=FOLD_KEYS, columns="representation", values="value"
        ).dropna()
        metric = block["metric"].iloc[0]

        for a, b in itertools.combinations(sorted(wide.columns), 2):
            diff = (wide[a] - wide[b]).to_numpy()
            if len(diff) < 2:
                continue
            t_stat, p_value = stats.ttest_rel(wide[a], wide[b])
            rows.append(
                {
                    "dataset": dataset,
                    "split": split,
                    "metric": metric,
                    "representation_a": DISPLAY_NAMES.get(a, a),
                    "representation_b": DISPLAY_NAMES.get(b, b),
                    "n_folds": len(diff),
                    "mean_delta": diff.mean(),
                    "std_delta": diff.std(ddof=1),
                    "t_statistic": float(t_stat),
                    "p_value": float(p_value),
                }
            )

    return pd.DataFrame(rows)


def split_effect(frame):
    """Compare the effect of the split with the spread between representations.

    For each representation the change caused by moving from the random
    split to the scaffold split is measured on matching seeds and folds.
    The final column reports, for the same dataset and split, how far apart
    the representations themselves are, so the two magnitudes can be read
    against each other.
    """
    rows = []
    for dataset, block in frame.groupby("dataset"):
        metric = block["metric"].iloc[0]

        wide = block.pivot_table(
            index=["representation", "seed", "fold"], columns="split", values="value"
        ).dropna()
        if not {"random", "scaffold"}.issubset(wide.columns):
            continue

        for representation, chunk in wide.groupby(level="representation"):
            delta = (chunk["scaffold"] - chunk["random"]).to_numpy()
            rows.append(
                {
                    "dataset": dataset,
                    "metric": metric,
                    "representation": DISPLAY_NAMES.get(representation, representation),
                    "n_folds": len(delta),
                    "random_mean": chunk["random"].mean(),
                    "scaffold_mean": chunk["scaffold"].mean(),
                    "split_delta": delta.mean(),
                    "split_delta_std": delta.std(ddof=1) if len(delta) > 1 else np.nan,
                }
            )

    table = pd.DataFrame(rows)
    if table.empty:
        return table

    # Spread between representations, for comparison with the split effect.
    for split in ("random", "scaffold"):
        means = (
            frame[frame["split"] == split]
            .groupby(["dataset", "representation"])["value"]
            .mean()
            .groupby("dataset")
        )
        spread = (means.max() - means.min()).rename(f"representation_spread_{split}")
        table = table.merge(spread, on="dataset", how="left")

    return table


def architecture_table(frame):
    """Input size, hidden width and parameter count per representation."""
    table = (
        frame.groupby(["dataset", "representation"])
        .agg(
            input_dim=("input_dim", "first"),
            hidden_width=("hidden_width", "first"),
            trainable_parameters=("trainable_parameters", "mean"),
            fit_seconds=("fit_seconds", "mean"),
            predict_seconds=("predict_seconds", "mean"),
        )
        .reset_index()
    )
    table["trainable_parameters"] = table["trainable_parameters"].round().astype(int)
    table["fit_seconds"] = table["fit_seconds"].round(1)
    table["predict_seconds"] = table["predict_seconds"].round(3)
    table["representation"] = table["representation"].map(DISPLAY_NAMES)
    return table


def to_latex(table, caption, label, float_format="%.3f"):
    """A LaTeX tabular that can be pasted into the manuscript."""
    body = table.to_latex(
        index=False, escape=True, float_format=float_format, na_rep="--"
    )
    return (
        "\\begin{table}[!t]\n\\centering\n\\footnotesize\n"
        f"\\caption{{{caption}}}\n\\label{{{label}}}\n"
        f"{body}"
        "\\end{table}\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results/per_fold_results.csv")
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--latex", action="store_true",
                        help="also write LaTeX versions of the tables")
    args = parser.parse_args()

    frame = load_results(args.results)
    os.makedirs(args.out_dir, exist_ok=True)

    outputs = {
        "main_results": main_results(frame),
        "paired_deltas": paired_deltas(frame),
        "split_effect": split_effect(frame),
        "architecture": architecture_table(frame),
    }

    for name, table in outputs.items():
        if table is None or table.empty:
            print(f"[report] {name}: nothing to write")
            continue
        path = os.path.join(args.out_dir, f"{name}.csv")
        table.to_csv(path, index=False)
        print(f"[report] wrote {path} ({len(table)} rows)")

    for split in sorted(frame["split"].unique()):
        table = results_table(frame, split)
        path = os.path.join(args.out_dir, f"table_{split}.csv")
        table.to_csv(path)
        print(f"[report] wrote {path}")
        print(f"\n--- {split} split ---")
        print(table.to_string())

    if args.latex:
        path = os.path.join(args.out_dir, "tables.tex")
        with open(path, "w") as handle:
            for split in sorted(frame["split"].unique()):
                handle.write(
                    to_latex(
                        results_table(frame, split).reset_index(),
                        caption=f"Test results under the {split} split, "
                                "reported as mean $\\pm$ standard deviation "
                                "across fifteen test folds.",
                        label=f"tab:results_{split}",
                    )
                )
                handle.write("\n")
            handle.write(
                to_latex(
                    outputs["architecture"],
                    caption="Architecture and trainable parameter count for each "
                            "representation.",
                    label="tab:architecture",
                )
            )
        print(f"[report] wrote {path}")


if __name__ == "__main__":
    main()
