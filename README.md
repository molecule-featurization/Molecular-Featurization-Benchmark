# Molecular Featurization Benchmark

Code for *Benchmarking Geometry-Based Molecular Representations for
Property Prediction*.

Five geometry-based molecular representations — Coulomb Matrix, Bag of
Bonds, ACSF, SOAP and Weighted Views — are compared on five MoleculeNet
datasets under a matched parameter budget, using both random and scaffold
cross-validation splits.

## Installation

```bash
pip install -r requirements.txt
```

DeepChem supplies the datasets. If it is not installed, the loader instead
reads a CSV from `data/<dataset>.csv` with a SMILES column and one column
per task.

## Running the benchmark

```bash
# one dataset, one split type
python -m pipeline.run --dataset bbbp --split random
python -m pipeline.run --dataset bbbp --split scaffold

# everything: five datasets, both split types
python -m pipeline.run --all

# then build the tables
python -m pipeline.report --latex
```

Defaults are five folds, three seeds, 150 epochs, batch size 32 and L2
0.01, which is the protocol described in the paper. Each run appends to
`results/per_fold_results.csv`; delete that file to start over.

Useful flags: `--seeds`, `--folds`, `--epochs`, `--representations`,
`--conformer-seed`.

## Layout

```
featurization/     the five representations
  coulomb_matrix.py
  bag_of_bonds.py
  acsf.py
  soap.py
  weightedviews.py

pipeline/
  datasets.py      loads the five MoleculeNet datasets
  geometry.py      SMILES to 3D coordinates, seeded and cached
  splits.py        random and Murcko scaffold cross-validation
  features.py      builds and caches each representation
  models.py        the two architectures and the parameter matching
  run.py           main loop, writes one row per fold
  report.py        per-fold rows to paper tables

results/           output
cache/             geometries and descriptors, safe to delete
```

## Outputs

`results/per_fold_results.csv` holds one row per fold with the metric, the
trainable parameter count, the hidden width, the fold sizes and the
timings. Everything else is derived from it by `pipeline/report.py`:

| File | Contents |
| --- | --- |
| `main_results.csv` | mean ± standard deviation per representation, dataset and split |
| `table_random.csv`, `table_scaffold.csv` | the same in the paper's layout |
| `paired_deltas.csv` | representation differences on matching folds, with paired t-tests |
| `split_effect.csv` | how far each representation moves when only the split changes, next to the spread between representations |
| `architecture.csv` | input dimension, hidden width, parameter count and timings |
| `dataset_summary.csv` | molecules excluded and why, final sample size, scaffold count, descriptor dimensions |
| `tables.tex` | LaTeX versions, with `--latex` |

## Notes on the method

**Reproducibility.** Conformer generation is seeded. The conformer seed is
separate from the cross-validation seeds, so every run sees the same
geometries and only the split and the weight initialisation change between
seeds.

**Descriptors.** All five representations are computed from the same set
of geometries, and every representation is evaluated on the same folds.
The settings are fixed and are not tuned per dataset or per representation.

**Pooling.** ACSF produces one vector per atom; the atomic vectors are
stacked, zero padded to the largest number of atoms in the training fold
and flattened. The same size is used for the test fold, so a test molecule
with more atoms than this is truncated. SOAP averages the atomic power
spectra over the molecule, through dscribe's `average="outer"`. The Coulomb
Matrix uses the sorted variant, with rows and columns permuted so the row
norms decrease, which makes it independent of the order of the atoms in
the input.

**Chemical species.** ACSF and SOAP use the same fixed list of 29 elements
for every dataset: H, B, C, N, O, F, Na, Al, Si, P, S, Cl, Ca, Ti, Cr, Mn,
Co, Cu, Zn, As, Se, Br, Tc, I, Pt, Au, Hg, Tl, Bi.

**Feature scaling.** The four flat representations are standardised
feature-wise, with the mean and variance taken from the training fold only.
Their natural scales differ by orders of magnitude — Coulomb Matrix entries
are Z_i Z_j / R and run to the hundreds, SOAP power spectrum entries are of
order one — so without this a representation is judged partly on its units.
Weighted Views is not scaled: its inputs are atomic coordinates and species
indicators, already on a common scale, and centring them would destroy the
meaning of the zero padding used for absent atoms. Pass `--no-standardize`
to turn it off.

**Parameter matching.** The Weighted Views model is built first and its
trainable parameters counted; the hidden width of the dense network used by
the other four is then chosen by binary search to land as close as possible
to that count. Because the width is an integer, the counts match
approximately rather than exactly, and the count actually achieved is
recorded for every fold in `per_fold_results.csv`.

**Hardware.** All experiments in the paper were run on an Apple MacBook Pro
(M2 Pro, 16 GB RAM), CPU only. The full set of experiments took about 17 to
18 hours.

**Splits.** The random split is stratified for classification and plain
k-fold for regression, as in the original submission. The scaffold split
groups molecules by Bemis-Murcko scaffold and assigns whole groups to
folds, largest group first, to whichever fold is currently smallest, so no
scaffold appears in both training and test. Different seeds break ties
differently and so give genuinely different scaffold partitions.

A molecule with no ring system has no Murcko scaffold, and RDKit returns an
empty string for it. On datasets that are largely small acyclic molecules,
such as ESOL and FreeSolv, treating that empty string as one scaffold puts
a large fraction of the data into a single group, and since a group is
never split the folds come out wildly uneven. Each acyclic molecule is
therefore given its own group, keyed by its canonical SMILES. Nothing leaks
by doing so: the point of a scaffold split is that a core seen in training
must not reappear in test, and a molecule with no ring system has no core
to share.

## Datasets

| Name | Task | Metric |
| --- | --- | --- |
| `esol` | regression | RMSE |
| `freesolv` | regression | RMSE |
| `bbbp` | classification | ROC-AUC |
| `bace` | classification | ROC-AUC |
| `clintox` | classification, 2 tasks | mean ROC-AUC |
