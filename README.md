
# Molecular-Featurization-Benchmark

This repository provides implementations of five widely used molecular featurization methods together with benchmark experiments on five MoleculeNet datasets. 

---

## Repository Contents

### Utility Files

- **smiles_to_mollist.py**  
  Converts molecular SMILES strings into three-dimensional molecular structures using RDKit. The generated molecular structures are used as input for all featurization methods.

- **coulomb_matrix.py**  
  Implementation of the Coulomb Matrix (CM) featurization.

- **bag_of_bonds.py**  
  Implementation of the Bag of Bonds (BoB) featurization.

- **acsf.py**  
  Implementation of Atom-Centered Symmetry Functions (ACSF).

- **soap.py**  
  Implementation of the Smooth Overlap of Atomic Positions (SOAP) descriptor.

- **weightedviews.py**  
  Implementation of the Weighted Views molecular representation.

---

### Experiment Notebooks

Each notebook evaluates all five molecular representations on a single benchmark dataset.

| Notebook | Dataset | Task |
|----------|---------|------|
| `esol_work.ipynb` | ESOL | Regression |
| `freesolv_work.ipynb` | FreeSolv | Regression |
| `bbbp_work.ipynb` | BBBP | Classification |
| `BACE_work.ipynb` | BACE | Classification |
| `ClinTox.ipynb` | ClinTox | Classification |

---

## Running the Experiments

Open any notebook and execute all cells.

For example,

- `esol_work.ipynb`
- `freesolv_work.ipynb`
- `bbbp_work.ipynb`
- `BACE_work.ipynb`
- `ClinTox.ipynb`

Each notebook automatically:

1. Downloads the corresponding MoleculeNet dataset.
2. Converts SMILES strings into three-dimensional molecular structures.
3. Generates all five molecular representations.
4. Trains the corresponding neural networks.
5. Reports the final test performance.

No manual dataset download is required.

---

## Molecular Representations

The repository includes implementations of the following molecular representations:

- Weighted Views
- Coulomb Matrix (CM)
- Bag of Bonds (BoB)
- Atom-Centered Symmetry Functions (ACSF)
- Smooth Overlap of Atomic Positions (SOAP)

---

