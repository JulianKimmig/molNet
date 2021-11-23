import numpy as np

from molNet.featurizer._atom_featurizer import FixedSizeAtomFeaturizer
from molNet.utils.mol import ATOMIC_SYMBOL_NUMBERS


class Atom_ConnectedAtoms_Featurizer(FixedSizeAtomFeaturizer):
    dtype = np.int32
    LENGTH = max(ATOMIC_SYMBOL_NUMBERS.values()) + 1  # +1 since 0  is possible
    atoms = list(ATOMIC_SYMBOL_NUMBERS.keys())

    def featurize(self, atom):
        connected_atom_types = np.zeros(self.LENGTH)
        for b in atom.GetBonds():
            connected_atom_types[b.GetOtherAtom(atom).GetAtomicNum()] += 1
        return connected_atom_types


atom_ConnectedAtoms_featurizer = Atom_ConnectedAtoms_Featurizer()
_available_featurizer = {
    "atom_ConnectedAtoms_featurizer": atom_ConnectedAtoms_featurizer,
}

__all__ = [
    "Atom_ConnectedAtoms_Featurizer",
    "atom_ConnectedAtoms_featurizer",
]


def get_available_featurizer():
    return _available_featurizer


def main():
    from rdkit import Chem

    testmol = Chem.MolFromSmiles("c1ccccc1").GetAtoms()[0]

    for n, f in get_available_featurizer().items():
        print(n, end=" ")
        print(f(testmol))
    print(len(get_available_featurizer()))
    print()


if __name__ == "__main__":
    main()
