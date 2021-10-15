import numpy as np
from rdkit.Chem.rdMolDescriptors import (
    GetHashedTopologicalTorsionFingerprint,
    GetHashedAtomPairFingerprintAsBitVect,
    GetHashedAtomPairFingerprint,
    GetHashedTopologicalTorsionFingerprintAsBitVect,
    GetMACCSKeysFingerprint,
)
from rdkit.Chem.rdmolops import (
    RDKFingerprint,
    LayeredFingerprint,
    PatternFingerprint,
)
from rdkit.DataStructs.cDataStructs import ConvertToNumpyArray

from molNet.featurizer._molecule_featurizer import (
    FixedSizeMoleculeFeaturizer,
)


class GetHashedAtomPairFingerprint_Featurizer(FixedSizeMoleculeFeaturizer):
    # statics
    LENGTH = 2048
    dtype = np.int32

    # normalization
    # functions
    def featurize(self, mol):
        a = np.zeros(len(self), dtype=self.dtype)
        ConvertToNumpyArray(GetHashedAtomPairFingerprint(mol), a)
        return a


class GetHashedAtomPairFingerprintAsBitVect_Featurizer(FixedSizeMoleculeFeaturizer):
    # statics
    LENGTH = 2048
    dtype = bool
    # normalization
    # functions
    def featurize(self, mol):
        a = np.zeros(len(self), dtype=self.dtype)
        ConvertToNumpyArray(GetHashedAtomPairFingerprintAsBitVect(mol), a)
        return a


class GetHashedTopologicalTorsionFingerprint_Featurizer(FixedSizeMoleculeFeaturizer):
    # statics
    LENGTH = 2048
    dtype = np.int64
    # normalization
    # functions
    def featurize(self, mol):
        a = np.zeros(len(self), dtype=self.dtype)
        ConvertToNumpyArray(GetHashedTopologicalTorsionFingerprint(mol), a)
        return a


class GetHashedTopologicalTorsionFingerprintAsBitVect_Featurizer(
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


class GetMACCSKeysFingerprint_Featurizer(FixedSizeMoleculeFeaturizer):
    # statics
    LENGTH = 167
    dtype = bool
    # normalization
    # functions
    def featurize(self, mol):
        a = np.zeros(len(self), dtype=self.dtype)
        ConvertToNumpyArray(GetMACCSKeysFingerprint(mol), a)
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


molecule_GetHashedAtomPairFingerprint = GetHashedAtomPairFingerprint_Featurizer()
molecule_GetHashedAtomPairFingerprintAsBitVect = (
    GetHashedAtomPairFingerprintAsBitVect_Featurizer()
)
molecule_GetHashedTopologicalTorsionFingerprint = (
    GetHashedTopologicalTorsionFingerprint_Featurizer()
)
molecule_GetHashedTopologicalTorsionFingerprintAsBitVect = (
    GetHashedTopologicalTorsionFingerprintAsBitVect_Featurizer()
)
molecule_GetMACCSKeysFingerprint = GetMACCSKeysFingerprint_Featurizer()
molecule_LayeredFingerprint = LayeredFingerprint_Featurizer()
molecule_PatternFingerprint = PatternFingerprint_Featurizer()
molecule_RDKFingerprint = RDKFingerprint_Featurizer()

_available_featurizer = [
    molecule_GetHashedAtomPairFingerprint,
    molecule_GetHashedAtomPairFingerprintAsBitVect,
    molecule_GetHashedTopologicalTorsionFingerprint,
    molecule_GetHashedTopologicalTorsionFingerprintAsBitVect,
    molecule_GetMACCSKeysFingerprint,
    molecule_LayeredFingerprint,
    molecule_PatternFingerprint,
    molecule_RDKFingerprint,
]


def main():
    from rdkit import Chem

    testmol = Chem.MolFromSmiles("c1ccccc1")
    for f in _available_featurizer.items():
        print(f, f(testmol))


if __name__ == "__main__":
    main()