import numpy as np
from ase import Atoms
from dscribe.descriptors import ACSF


def create_acsf(
    molecule_list,
    max_atoms=None,
    species=[
        "H", "B", "C", "N", "O", "F",
        "Na",
        "Al", "Si", "P", "S", "Cl",
        "Ca", "Ti", "Cr", "Mn", "Co", "Cu", "Zn",
        "As", "Se", "Br",
        "Tc",
        "I",
        "Pt", "Au", "Hg", "Tl", "Bi",
    ],
    r_cut=6.0,
):
    """
    Generate ACSF descriptors.

    Parameters
    ----------
    molecule_list : list
        List of molecules.

    max_atoms : int or None
        Maximum number of atoms.
        If None, determine from the current dataset.
        If provided, pad/truncate to this size.

    Returns
    -------
    X : np.ndarray
        Shape = (n_molecules, max_atoms, n_features)

    max_atoms : int
        Maximum number of atoms used.
    """

    ase_molecules = []

    for molecule in molecule_list:

        symbols = [atom[0] for atom in molecule]
        positions = [atom[1] for atom in molecule]

        ase_molecules.append(
            Atoms(symbols=symbols, positions=positions)
        )

    acsf = ACSF(
        species=species,
        r_cut=r_cut,
        g2_params=[
            [1, 1],
            [1, 2],
            [1, 3],
        ],
        g4_params=[
            [1, 1, 1],
            [1, 2, 1],
            [1, 1, -1],
            [1, 2, -1],
        ],
    )

    features = [acsf.create(mol) for mol in ase_molecules]

    if max_atoms is None:
        max_atoms = max(x.shape[0] for x in features)

    n_features = features[0].shape[1]

    X = np.zeros(
        (len(features), max_atoms, n_features),
        dtype=np.float32,
    )

    for i, x in enumerate(features):

        n = min(x.shape[0], max_atoms)

        X[i, :n, :] = x[:n]

    return X, max_atoms