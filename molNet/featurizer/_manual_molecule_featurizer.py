import numpy as np
from rdkit.Chem import GetMolFrags

from molNet.featurizer._molecule_featurizer import MoleculeFeaturizer, SingleValueMoleculeFeaturizer


class ExtendMolnetFeaturizer(MoleculeFeaturizer):
    def featurize(self, mol):
        r = mol.molnet_features if hasattr(mol, "molnet_features") else []
        return np.array([r]).flatten()


extend_molnet_featurizer = ExtendMolnetFeaturizer(name="extend_molnet_featurizer")


class NumFragments_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.int32
    # functions

    def featurize(self, mol):
        return len(GetMolFrags(mol))


molecule_num_fragments = NumFragments_Featurizer()

_available_featurizer = {
    "molecule_num_fragments": molecule_num_fragments,
}


def get_available_featurizer():
    return _available_featurizer


__all__ = [
    "NumFragments_Featurizer",
    "molecule_num_fragments",

]


def main():
    from rdkit import Chem
    from molNet.featurizer.molecule_featurizer import prepare_mol_for_featurization

    testmol = prepare_mol_for_featurization(Chem.MolFromSmiles("c1ccccc1"))
    for n, f in get_available_featurizer().items():
        print(n, end=" ")
        print(f(testmol))
    print(len(get_available_featurizer()))

if __name__ == "__main__":
    main()
