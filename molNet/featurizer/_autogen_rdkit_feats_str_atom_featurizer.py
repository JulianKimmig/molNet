import numpy as np
from molNet.featurizer._atom_featurizer import StringAtomFeaturizer


class Atom_Smarts_Featurizer(StringAtomFeaturizer):
    # _rdfunc=GetSmarts
    dtype = str

    def featurize(self, atom):
        return atom.GetSmarts()


class Atom_Symbol_Featurizer(StringAtomFeaturizer):
    # _rdfunc=GetSymbol
    dtype = str

    def featurize(self, atom):
        return atom.GetSymbol()


atom_Smarts_featurizer = Atom_Smarts_Featurizer()
atom_Symbol_featurizer = Atom_Symbol_Featurizer()
_available_featurizer = {
    "atom_Smarts_featurizer": atom_Smarts_featurizer,
    "atom_Symbol_featurizer": atom_Symbol_featurizer,
}
__all__ = [
    "Atom_Smarts_Featurizer",
    "atom_Smarts_featurizer",
    "Atom_Symbol_Featurizer",
    "atom_Symbol_featurizer",
]


def get_available_featurizer():
    return _available_featurizer


def main():
    from rdkit import Chem
    from molNet.featurizer.molecule_featurizer import prepare_mol_for_featurization

    testdata = prepare_mol_for_featurization(Chem.MolFromSmiles("c1ccccc1")).GetAtoms()[
        -1
    ]
    for n, f in get_available_featurizer().items():
        print(n, end=" ")
        print(f(testdata))
    print(len(get_available_featurizer()))


if __name__ == "__main__":
    main()
