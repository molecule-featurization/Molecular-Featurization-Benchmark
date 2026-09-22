"""
coulomb_matrix.py

Generic Coulomb Matrix generator for molecular datasets.

Input
-----
mollist : list
    [
        [
            ('C', np.array([x, y, z])),
            ('O', np.array([x, y, z])),
            ...
        ],
        ...
    ]

Output
------
numpy.ndarray
    Shape = (n_molecules, max_atoms, max_atoms)
"""

import numpy as np
from rdkit.Chem import GetPeriodicTable


ptable = GetPeriodicTable()


def create_coulomb_matrix(mollist, max_atoms=None, sort=True):
    """
    Create Coulomb matrices for a list of molecules.

    Parameters
    ----------
    mollist : list
        Molecular data in the form
        [('C', xyz), ('O', xyz), ...]

    max_atoms : int or None
        Maximum number of atoms.
        If None, determined automatically from mollist.

    sort : bool
        If True, rows and columns are permuted so that the row norms are in
        descending order.  This is the sorted variant of the Coulomb Matrix
        and is what makes the representation invariant to the ordering of
        the atoms in the input.  The unsorted matrix depends on atom order
        and is not a well defined molecular descriptor, so this defaults to
        True.

    Returns
    -------
    numpy.ndarray
        Coulomb matrices with shape
        (n_molecules, max_atoms, max_atoms)
    """

    if max_atoms is None:
        max_atoms = max(len(mol) for mol in mollist)

    CM_all = np.zeros(
        (len(mollist), max_atoms, max_atoms),
        dtype=np.float32
    )

    for m_idx, molecule in enumerate(mollist):

        n_atoms = len(molecule)

        for i in range(n_atoms):

            symbol_i, coord_i = molecule[i]
            Zi = ptable.GetAtomicNumber(symbol_i)

            for j in range(n_atoms):

                symbol_j, coord_j = molecule[j]
                Zj = ptable.GetAtomicNumber(symbol_j)

                if i == j:

                    CM_all[m_idx, i, i] = 0.5 * Zi ** 2.4

                else:

                    Rij = np.linalg.norm(coord_i - coord_j)

                    CM_all[m_idx, i, j] = Zi * Zj / Rij

        if sort:
            CM_all[m_idx] = _sort_by_row_norm(CM_all[m_idx])

    return CM_all


def _sort_by_row_norm(matrix):
    """Permute rows and columns so the row norms decrease down the matrix.

    The same permutation is applied to the rows and to the columns, so the
    result stays symmetric and still describes the same molecule.
    """
    order = np.argsort(-np.linalg.norm(matrix, axis=1), kind="stable")
    return matrix[order][:, order]