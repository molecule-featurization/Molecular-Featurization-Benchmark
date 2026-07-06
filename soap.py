import numpy as np
from ase import Atoms
from dscribe.descriptors import SOAP


def create_soap(
    molecule_list,
 species=[
    "H", "B", "C", "N", "O", "F",
    "Na",
    "Al", "Si", "P", "S", "Cl",
    "Ca", "Ti", "Cr", "Mn", "Co", "Cu", "Zn",
    "As", "Se", "Br",
    "Tc",
    "I",
    "Pt", "Au", "Hg", "Tl", "Bi"
],
    r_cut=6.0,
    n_max=8,
    l_max=6,
    sigma=1.0,
):
    """
    Generate SOAP descriptors.

    Parameters
    ----------
    molecule_list : list
        List of molecules in the format:
        [
            [
                ('C', np.array([x, y, z])),
                ('H', np.array([x, y, z])),
                ...
            ],
            ...
        ]

    Returns
    -------
    numpy.ndarray
        Array of shape (n_molecules, n_features)
    """

    ase_molecules = []

    for molecule in molecule_list:
        symbols = [atom[0] for atom in molecule]
        positions = [atom[1] for atom in molecule]

        ase_molecules.append(
            Atoms(symbols=symbols, positions=positions)
        )

    soap = SOAP(
        species=species,
        r_cut=r_cut,
        n_max=n_max,
        l_max=l_max,
        sigma=sigma,
        rbf="gto",
        average="outer",
        periodic=False,
        sparse=False,
    )

    X = np.array(
        [soap.create(mol) for mol in ase_molecules],
        dtype=np.float32,
    )

    return X