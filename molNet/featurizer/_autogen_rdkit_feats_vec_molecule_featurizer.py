import numpy as np
from molNet.featurizer._molecule_featurizer import (
    VarSizeMoleculeFeaturizer,
    FixedSizeMoleculeFeaturizer,
)
from rdkit.DataStructs import ConvertToNumpyArray
from rdkit.Chem import PatternFingerprint, LayeredFingerprint, RDKFingerprint
from rdkit.Chem.AllChem import (
    GetHashedAtomPairFingerprint,
    GetMACCSKeysFingerprint,
    GetHashedTopologicalTorsionFingerprint,
    GetHashedAtomPairFingerprintAsBitVect,
    GetHashedTopologicalTorsionFingerprintAsBitVect,
)


class Molecule_RDKFingerprint_Featurizer(FixedSizeMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.RDKFingerprint
    dtype = bool
    LENGTH = 2048

    def featurize(self, mol):
        a = np.zeros(len(self), dtype=self.dtype)
        ConvertToNumpyArray(RDKFingerprint(mol), a)
        return a


class Molecule_HashedAtomPairFingerprint_Featurizer(FixedSizeMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.AllChem.GetHashedAtomPairFingerprint
    dtype = np.int32
    LENGTH = 2048

    def featurize(self, mol):
        a = np.zeros(len(self), dtype=self.dtype)
        ConvertToNumpyArray(GetHashedAtomPairFingerprint(mol), a)
        return a


class Molecule_HashedTopologicalTorsionFingerprintAsBitVect_Featurizer(
    FixedSizeMoleculeFeaturizer
):
    # _rdfunc=rdkit.Chem.AllChem.GetHashedTopologicalTorsionFingerprintAsBitVect
    dtype = bool
    LENGTH = 2048

    def featurize(self, mol):
        a = np.zeros(len(self), dtype=self.dtype)
        ConvertToNumpyArray(GetHashedTopologicalTorsionFingerprintAsBitVect(mol), a)
        return a


class Molecule_HashedTopologicalTorsionFingerprint_Featurizer(
    FixedSizeMoleculeFeaturizer
):
    # _rdfunc=rdkit.Chem.AllChem.GetHashedTopologicalTorsionFingerprint
    dtype = np.int64
    LENGTH = 2048

    def featurize(self, mol):
        a = np.zeros(len(self), dtype=self.dtype)
        ConvertToNumpyArray(GetHashedTopologicalTorsionFingerprint(mol), a)
        return a


class Molecule_LayeredFingerprint_Featurizer(FixedSizeMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.LayeredFingerprint
    dtype = bool
    LENGTH = 2048

    def featurize(self, mol):
        a = np.zeros(len(self), dtype=self.dtype)
        ConvertToNumpyArray(LayeredFingerprint(mol), a)
        return a


class Molecule_MACCSKeysFingerprint_Featurizer(FixedSizeMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.AllChem.GetMACCSKeysFingerprint
    dtype = bool
    LENGTH = 167

    def featurize(self, mol):
        a = np.zeros(len(self), dtype=self.dtype)
        ConvertToNumpyArray(GetMACCSKeysFingerprint(mol), a)
        return a


class Molecule_PatternFingerprint_Featurizer(FixedSizeMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.PatternFingerprint
    dtype = bool
    LENGTH = 2048

    def featurize(self, mol):
        a = np.zeros(len(self), dtype=self.dtype)
        ConvertToNumpyArray(PatternFingerprint(mol), a)
        return a


class Molecule_HashedAtomPairFingerprintAsBitVect_Featurizer(
    FixedSizeMoleculeFeaturizer
):
    # _rdfunc=rdkit.Chem.AllChem.GetHashedAtomPairFingerprintAsBitVect
    dtype = bool
    LENGTH = 2048

    def featurize(self, mol):
        a = np.zeros(len(self), dtype=self.dtype)
        ConvertToNumpyArray(GetHashedAtomPairFingerprintAsBitVect(mol), a)
        return a


molecule_RDKFingerprint_featurizer = Molecule_RDKFingerprint_Featurizer()
molecule_HashedAtomPairFingerprint_featurizer = (
    Molecule_HashedAtomPairFingerprint_Featurizer()
)
molecule_HashedTopologicalTorsionFingerprintAsBitVect_featurizer = (
    Molecule_HashedTopologicalTorsionFingerprintAsBitVect_Featurizer()
)
molecule_HashedTopologicalTorsionFingerprint_featurizer = (
    Molecule_HashedTopologicalTorsionFingerprint_Featurizer()
)
molecule_LayeredFingerprint_featurizer = Molecule_LayeredFingerprint_Featurizer()
molecule_MACCSKeysFingerprint_featurizer = Molecule_MACCSKeysFingerprint_Featurizer()
molecule_PatternFingerprint_featurizer = Molecule_PatternFingerprint_Featurizer()
molecule_HashedAtomPairFingerprintAsBitVect_featurizer = (
    Molecule_HashedAtomPairFingerprintAsBitVect_Featurizer()
)
_available_featurizer = {
    "molecule_RDKFingerprint_featurizer": molecule_RDKFingerprint_featurizer,
    "molecule_HashedAtomPairFingerprint_featurizer": molecule_HashedAtomPairFingerprint_featurizer,
    "molecule_HashedTopologicalTorsionFingerprintAsBitVect_featurizer": molecule_HashedTopologicalTorsionFingerprintAsBitVect_featurizer,
    "molecule_HashedTopologicalTorsionFingerprint_featurizer": molecule_HashedTopologicalTorsionFingerprint_featurizer,
    "molecule_LayeredFingerprint_featurizer": molecule_LayeredFingerprint_featurizer,
    "molecule_MACCSKeysFingerprint_featurizer": molecule_MACCSKeysFingerprint_featurizer,
    "molecule_PatternFingerprint_featurizer": molecule_PatternFingerprint_featurizer,
    "molecule_HashedAtomPairFingerprintAsBitVect_featurizer": molecule_HashedAtomPairFingerprintAsBitVect_featurizer,
}
__all__ = [
    "Molecule_RDKFingerprint_Featurizer",
    "molecule_RDKFingerprint_featurizer",
    "Molecule_HashedAtomPairFingerprint_Featurizer",
    "molecule_HashedAtomPairFingerprint_featurizer",
    "Molecule_HashedTopologicalTorsionFingerprintAsBitVect_Featurizer",
    "molecule_HashedTopologicalTorsionFingerprintAsBitVect_featurizer",
    "Molecule_HashedTopologicalTorsionFingerprint_Featurizer",
    "molecule_HashedTopologicalTorsionFingerprint_featurizer",
    "Molecule_LayeredFingerprint_Featurizer",
    "molecule_LayeredFingerprint_featurizer",
    "Molecule_MACCSKeysFingerprint_Featurizer",
    "molecule_MACCSKeysFingerprint_featurizer",
    "Molecule_PatternFingerprint_Featurizer",
    "molecule_PatternFingerprint_featurizer",
    "Molecule_HashedAtomPairFingerprintAsBitVect_Featurizer",
    "molecule_HashedAtomPairFingerprintAsBitVect_featurizer",
]


def get_available_featurizer():
    return _available_featurizer


def main():
    from rdkit import Chem
    from molNet.featurizer.molecule_featurizer import prepare_mol_for_featurization

    testdata = prepare_mol_for_featurization(Chem.MolFromSmiles("c1ccccc1"))
    for n, f in get_available_featurizer().items():
        print(n, end=" ")
        print(f(testdata))
    print(len(get_available_featurizer()))


if __name__ == "__main__":
    main()
