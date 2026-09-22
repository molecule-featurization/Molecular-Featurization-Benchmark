import numpy as np
from collections import defaultdict


def create_bob(molecule_list, bag_sizes=None):
    """
    Create Bag of Bonds descriptors.

    Parameters
    ----------
    molecule_list : list
        List of molecules:
        [
            [
                ('C', np.array([x,y,z])),
                ('H', np.array([x,y,z])),
                ...
            ],
            ...
        ]

    bag_sizes : dict or None
        Dictionary of maximum bag sizes learned from the training set.
        Use None when processing the training set.

    Returns
    -------
    X : np.ndarray
        Bag of Bonds feature matrix.

    bag_sizes : dict
        Maximum size of each bag.
    """

    all_bags = []
    max_sizes = defaultdict(int)

    # ---------- Build bags ----------
    for molecule in molecule_list:

        bags = defaultdict(list)

        n = len(molecule)

        for i in range(n):

            Zi = molecule[i][0]
            Ri = molecule[i][1]

            for j in range(i + 1, n):

                Zj = molecule[j][0]
                Rj = molecule[j][1]

                key = tuple(sorted((Zi, Zj)))

                dist = np.linalg.norm(Ri - Rj)

                value = (
                    atomic_numbers[Zi]
                    * atomic_numbers[Zj]
                    / dist
                )

                bags[key].append(value)

        # sort each bag
        for key in bags:
            bags[key].sort(reverse=True)

        all_bags.append(bags)

        if bag_sizes is None:
            for key in bags:
                max_sizes[key] = max(max_sizes[key], len(bags[key]))

    if bag_sizes is None:
        bag_sizes = dict(max_sizes)

    # ---------- Build feature vectors ----------
    feature_vectors = []

    ordered_keys = sorted(bag_sizes.keys())

    for bags in all_bags:

        feature = []

        for key in ordered_keys:

            values = bags.get(key, [])

            values = values[:bag_sizes[key]]

            values += [0.0] * (bag_sizes[key] - len(values))

            feature.extend(values)

        feature_vectors.append(feature)

    return np.array(feature_vectors, dtype=np.float32), bag_sizes


atomic_numbers = {
    "H": 1,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Na": 11,
    "Al": 13,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "K": 19,
    "Ca": 20,
    "Ti": 22,
    "Cr": 24,
    "Mn": 25,
    "Co": 27,
    "Cu": 29,
    "Zn": 30,
    "As": 33,
    "Se": 34,
    "Br": 35,
    "Tc": 43,
    "I": 53,
    "Pt": 78,
    "Au": 79,
    "Hg": 80,
    "Tl": 81,
    "Bi": 83,
}