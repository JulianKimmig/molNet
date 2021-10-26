import numpy as np
from rdkit.Chem.rdMolDescriptors import (
    GetHashedAtomPairFingerprintAsBitVect,
    GetHashedAtomPairFingerprint,
    GetMACCSKeysFingerprint,
    GetHashedTopologicalTorsionFingerprint,
    GetHashedTopologicalTorsionFingerprintAsBitVect,
)
from rdkit.Chem.rdmolops import (
    LayeredFingerprint,
    PatternFingerprint,
    RDKFingerprint,
)
from rdkit.DataStructs.cDataStructs import ConvertToNumpyArray

from molNet.featurizer._molecule_featurizer import (
    FixedSizeMoleculeFeaturizer,
)


class HashedAtomPairFingerprint_Featurizer(FixedSizeMoleculeFeaturizer):
    # statics
    LENGTH = 2048
    dtype = np.int32

    # normalization
    # functions
    def featurize(self, mol):
        a = np.zeros(len(self), dtype=self.dtype)
        ConvertToNumpyArray(GetHashedAtomPairFingerprint(mol), a)
        return a


class HashedAtomPairFingerprintAsBitVect_Featurizer(FixedSizeMoleculeFeaturizer):
    # statics
    LENGTH = 2048
    dtype = bool

    # normalization
    # functions
    def featurize(self, mol):
        a = np.zeros(len(self), dtype=self.dtype)
        ConvertToNumpyArray(GetHashedAtomPairFingerprintAsBitVect(mol), a)
        return a


class HashedTopologicalTorsionFingerprint_Featurizer(FixedSizeMoleculeFeaturizer):
    # statics
    LENGTH = 2048
    dtype = np.int64

    # normalization
    # functions
    def featurize(self, mol):
        a = np.zeros(len(self), dtype=self.dtype)
        ConvertToNumpyArray(GetHashedTopologicalTorsionFingerprint(mol), a)
        return a


class HashedTopologicalTorsionFingerprintAsBitVect_Featurizer(
    FixedSizeMoleculeFeaturizer
):
    # statics
    LENGTH = 2048
    dtype = bool

    # normalization
    # functions
    def featurize(self, mol):
        a = np.zeros(len(self), dtype=self.dtype)
        ConvertToNumpyArray(GetHashedTopologicalTorsionFingerprintAsBitVect(mol), a)
        return a


class LayeredFingerprint_Featurizer(FixedSizeMoleculeFeaturizer):
    # statics
    LENGTH = 2048
    dtype = bool

    # normalization
    # functions
    def featurize(self, mol):
        a = np.zeros(len(self), dtype=self.dtype)
        ConvertToNumpyArray(LayeredFingerprint(mol), a)
        return a


class MACCSKeysFingerprint_Featurizer(FixedSizeMoleculeFeaturizer):
    # statics
    LENGTH = 167
    dtype = bool

    # normalization
    # functions
    def featurize(self, mol):
        a = np.zeros(len(self), dtype=self.dtype)
        ConvertToNumpyArray(GetMACCSKeysFingerprint(mol), a)
        return a


class PatternFingerprint_Featurizer(FixedSizeMoleculeFeaturizer):
    # statics
    LENGTH = 2048
    dtype = bool
    # normalization
    # functions
    def featurize(self, mol):
        a = np.zeros(len(self), dtype=self.dtype)
        ConvertToNumpyArray(PatternFingerprint(mol), a)
        return a


class RDKFingerprint_Featurizer(FixedSizeMoleculeFeaturizer):
    # statics
    LENGTH = 2048
    dtype = bool

    # normalization
    # functions
    def featurize(self, mol):
        a = np.zeros(len(self), dtype=self.dtype)
        ConvertToNumpyArray(RDKFingerprint(mol), a)
        return a


molecule_HashedAtomPairFingerprint = HashedAtomPairFingerprint_Featurizer()
molecule_HashedAtomPairFingerprintAsBitVect = (
    HashedAtomPairFingerprintAsBitVect_Featurizer()
)
molecule_HashedTopologicalTorsionFingerprint = (
    HashedTopologicalTorsionFingerprint_Featurizer()
)
molecule_HashedTopologicalTorsionFingerprintAsBitVect = (
    HashedTopologicalTorsionFingerprintAsBitVect_Featurizer()
)
molecule_LayeredFingerprint = LayeredFingerprint_Featurizer()
molecule_MACCSKeysFingerprint = MACCSKeysFingerprint_Featurizer()
molecule_PatternFingerprint = PatternFingerprint_Featurizer()
molecule_RDKFingerprint = RDKFingerprint_Featurizer()

_available_featurizer = {
    "molecule_HashedAtomPairFingerprint": molecule_HashedAtomPairFingerprint,
    "molecule_HashedAtomPairFingerprintAsBitVect": molecule_HashedAtomPairFingerprintAsBitVect,
    "molecule_HashedTopologicalTorsionFingerprint": molecule_HashedTopologicalTorsionFingerprint,
    "molecule_HashedTopologicalTorsionFingerprintAsBitVect": molecule_HashedTopologicalTorsionFingerprintAsBitVect,
    "molecule_LayeredFingerprint": molecule_LayeredFingerprint,
    "molecule_MACCSKeysFingerprint": molecule_MACCSKeysFingerprint,
    "molecule_PatternFingerprint": molecule_PatternFingerprint,
    "molecule_RDKFingerprint": molecule_RDKFingerprint,
}


def get_available_featurizer():
    return _available_featurizer


__all__ = [
    "HashedAtomPairFingerprint_Featurizer",
    "HashedAtomPairFingerprintAsBitVect_Featurizer",
    "HashedTopologicalTorsionFingerprint_Featurizer",
    "HashedTopologicalTorsionFingerprintAsBitVect_Featurizer",
    "LayeredFingerprint_Featurizer",
    "MACCSKeysFingerprint_Featurizer",
    "PatternFingerprint_Featurizer",
    "RDKFingerprint_Featurizer",
    "molecule_HashedAtomPairFingerprint",
    "molecule_HashedAtomPairFingerprintAsBitVect",
    "molecule_HashedTopologicalTorsionFingerprint",
    "molecule_HashedTopologicalTorsionFingerprintAsBitVect",
    "molecule_LayeredFingerprint",
    "molecule_MACCSKeysFingerprint",
    "molecule_PatternFingerprint",
    "molecule_RDKFingerprint",
]


def main():
    from rdkit import Chem

    testmol = Chem.MolFromSmiles("c1ccccc1")
    for n, f in get_available_featurizer().items():
        print(n, f(testmol))


if __name__ == "__main__":
    main()
