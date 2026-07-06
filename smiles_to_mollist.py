import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem


def smiles_to_mollist(smiles_list):
    """
    Convert a list of SMILES into the mollist format.

    Returns
    -------
    mollist : list
        [
            [
                ('C', np.array([x, y, z])),
                ('H', np.array([x, y, z])),
                ...
            ],
            ...
        ]
    """

    mollist = []

    for smiles in smiles_list:

        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            continue

        mol = Chem.AddHs(mol)

        status = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())

        if status != 0:
            continue

        AllChem.UFFOptimizeMolecule(mol)

        conf = mol.GetConformer()

        molecule = []

        for atom in mol.GetAtoms():

            pos = conf.GetAtomPosition(atom.GetIdx())

            molecule.append(
                (
                    atom.GetSymbol(),
                    np.array([pos.x, pos.y, pos.z], dtype=np.float32),
                )
            )

        mollist.append(molecule)

    return mollist