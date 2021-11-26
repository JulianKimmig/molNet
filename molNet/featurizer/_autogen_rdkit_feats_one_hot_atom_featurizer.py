import numpy as np
from molNet.featurizer._atom_featurizer import OneHotAtomFeaturizer
from rdkit.Chem.rdchem import HybridizationType, ChiralType


class Atom_ChiralTag_Featurizer(OneHotAtomFeaturizer):
    # _rdfunc=GetChiralTag
    POSSIBLE_VALUES = [
        ChiralType.CHI_UNSPECIFIED,
        ChiralType.CHI_TETRAHEDRAL_CW,
        ChiralType.CHI_TETRAHEDRAL_CCW,
        ChiralType.CHI_OTHER,
    ]

    def featurize(self, atom):
        return atom.GetChiralTag()


class Atom_Hybridization_Featurizer(OneHotAtomFeaturizer):
    # _rdfunc=GetHybridization
    POSSIBLE_VALUES = [
        HybridizationType.UNSPECIFIED,
        HybridizationType.S,
        HybridizationType.SP,
        HybridizationType.SP2,
        HybridizationType.SP3,
        HybridizationType.SP3D,
        HybridizationType.SP3D2,
        HybridizationType.OTHER,
    ]

    def featurize(self, atom):
        return atom.GetHybridization()


atom_ChiralTag_featurizer = Atom_ChiralTag_Featurizer()
atom_Hybridization_featurizer = Atom_Hybridization_Featurizer()
_available_featurizer = {
    "atom_ChiralTag_featurizer": atom_ChiralTag_featurizer,
    "atom_Hybridization_featurizer": atom_Hybridization_featurizer,
}
__all__ = [
    "Atom_ChiralTag_Featurizer",
    "atom_ChiralTag_featurizer",
    "Atom_Hybridization_Featurizer",
    "atom_Hybridization_featurizer",
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
