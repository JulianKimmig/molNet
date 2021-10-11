from molNet.featurizer._molecule_featurizer import (
    MoleculeFeaturizer,
    SingleValueMoleculeFeaturizer,
)
from molNet.featurizer.featurizer import FixedSizeFeaturizer
import numpy as np
from numpy import inf, nan
from rdkit.DataStructs.cDataStructs import ConvertToNumpyArray
from rdkit.Chem.rdMolDescriptors import (
<<<<<<< HEAD
    GetHashedAtomPairFingerprintAsBitVect,
    GetHashedTopologicalTorsionFingerprint,
    GetHashedTopologicalTorsionFingerprintAsBitVect,
    GetHashedAtomPairFingerprint,
    GetMACCSKeysFingerprint,
=======
    GetHashedTopologicalTorsionFingerprint,
    GetMACCSKeysFingerprint,
    GetHashedTopologicalTorsionFingerprintAsBitVect,
    GetHashedAtomPairFingerprintAsBitVect,
    GetHashedAtomPairFingerprint,
>>>>>>> 3a4ef9b47d39261f7b23c48e5302154b6b288fc7
)
from rdkit.Chem.rdmolops import (
    PatternFingerprint,
    LayeredFingerprint,
    RDKFingerprint,
)


<<<<<<< HEAD
class GetHashedAtomPairFingerprint_Featurizer(FixedSizeFeaturizer, MoleculeFeaturizer):
=======
class GetHashedAtomPairFingerprintAsBitVect_Featurizer(
    FixedSizeFeaturizer, MoleculeFeaturizer
):
>>>>>>> 3a4ef9b47d39261f7b23c48e5302154b6b288fc7
    # statics
    LENGTH = 2048
    dtype = np.int32
    # normalization
    # functions
    def featurize(self, mol):
        a = np.zeros(len(self), dtype=self.dtype)
<<<<<<< HEAD
        ConvertToNumpyArray(GetHashedAtomPairFingerprint(mol), a)
        return a


class GetHashedAtomPairFingerprintAsBitVect_Featurizer(
    FixedSizeFeaturizer, MoleculeFeaturizer
):
=======
        ConvertToNumpyArray(GetHashedAtomPairFingerprintAsBitVect(mol), a)
        return a


class PatternFingerprint_Featurizer(FixedSizeFeaturizer, MoleculeFeaturizer):
>>>>>>> 3a4ef9b47d39261f7b23c48e5302154b6b288fc7
    # statics
    LENGTH = 2048
    dtype = np.int32
    # normalization
    # functions
    def featurize(self, mol):
        a = np.zeros(len(self), dtype=self.dtype)
<<<<<<< HEAD
        ConvertToNumpyArray(GetHashedAtomPairFingerprintAsBitVect(mol), a)
        return a


class GetHashedTopologicalTorsionFingerprint_Featurizer(
    FixedSizeFeaturizer, MoleculeFeaturizer
):
    # statics
    LENGTH = 2048
    dtype = np.int64
=======
        ConvertToNumpyArray(PatternFingerprint(mol), a)
        return a


class GetMACCSKeysFingerprint_Featurizer(FixedSizeFeaturizer, MoleculeFeaturizer):
    # statics
    LENGTH = 167
    dtype = np.int32
>>>>>>> 3a4ef9b47d39261f7b23c48e5302154b6b288fc7
    # normalization
    # functions
    def featurize(self, mol):
        a = np.zeros(len(self), dtype=self.dtype)
<<<<<<< HEAD
        ConvertToNumpyArray(GetHashedTopologicalTorsionFingerprint(mol), a)
        return a


class GetHashedTopologicalTorsionFingerprintAsBitVect_Featurizer(
    FixedSizeFeaturizer, MoleculeFeaturizer
):
=======
        ConvertToNumpyArray(GetMACCSKeysFingerprint(mol), a)
        return a


class RDKFingerprint_Featurizer(FixedSizeFeaturizer, MoleculeFeaturizer):
>>>>>>> 3a4ef9b47d39261f7b23c48e5302154b6b288fc7
    # statics
    LENGTH = 2048
    dtype = np.int32
    # normalization
    # functions
    def featurize(self, mol):
        a = np.zeros(len(self), dtype=self.dtype)
<<<<<<< HEAD
        ConvertToNumpyArray(GetHashedTopologicalTorsionFingerprintAsBitVect(mol), a)
        return a


class GetMACCSKeysFingerprint_Featurizer(FixedSizeFeaturizer, MoleculeFeaturizer):
    # statics
    LENGTH = 167
    dtype = np.int32
=======
        ConvertToNumpyArray(RDKFingerprint(mol), a)
        return a


class GetHashedTopologicalTorsionFingerprint_Featurizer(
    FixedSizeFeaturizer, MoleculeFeaturizer
):
    # statics
    LENGTH = 2048
    dtype = np.int64
>>>>>>> 3a4ef9b47d39261f7b23c48e5302154b6b288fc7
    # normalization
    # functions
    def featurize(self, mol):
        a = np.zeros(len(self), dtype=self.dtype)
<<<<<<< HEAD
        ConvertToNumpyArray(GetMACCSKeysFingerprint(mol), a)
=======
        ConvertToNumpyArray(GetHashedTopologicalTorsionFingerprint(mol), a)
>>>>>>> 3a4ef9b47d39261f7b23c48e5302154b6b288fc7
        return a


class LayeredFingerprint_Featurizer(FixedSizeFeaturizer, MoleculeFeaturizer):
    # statics
    LENGTH = 2048
    dtype = np.int32
    # normalization
    # functions
    def featurize(self, mol):
        a = np.zeros(len(self), dtype=self.dtype)
        ConvertToNumpyArray(LayeredFingerprint(mol), a)
        return a


<<<<<<< HEAD
class PatternFingerprint_Featurizer(FixedSizeFeaturizer, MoleculeFeaturizer):
=======
class GetHashedAtomPairFingerprint_Featurizer(FixedSizeFeaturizer, MoleculeFeaturizer):
>>>>>>> 3a4ef9b47d39261f7b23c48e5302154b6b288fc7
    # statics
    LENGTH = 2048
    dtype = np.int32
    # normalization
    # functions
    def featurize(self, mol):
        a = np.zeros(len(self), dtype=self.dtype)
<<<<<<< HEAD
        ConvertToNumpyArray(PatternFingerprint(mol), a)
        return a


class RDKFingerprint_Featurizer(FixedSizeFeaturizer, MoleculeFeaturizer):
=======
        ConvertToNumpyArray(GetHashedAtomPairFingerprint(mol), a)
        return a


class GetHashedTopologicalTorsionFingerprintAsBitVect_Featurizer(
    FixedSizeFeaturizer, MoleculeFeaturizer
):
>>>>>>> 3a4ef9b47d39261f7b23c48e5302154b6b288fc7
    # statics
    LENGTH = 2048
    dtype = np.int32
    # normalization
    # functions
    def featurize(self, mol):
        a = np.zeros(len(self), dtype=self.dtype)
<<<<<<< HEAD
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

_available_featurizer = {
    "molecule_GetHashedAtomPairFingerprint": molecule_GetHashedAtomPairFingerprint,
    "molecule_GetHashedAtomPairFingerprintAsBitVect": molecule_GetHashedAtomPairFingerprintAsBitVect,
    "molecule_GetHashedTopologicalTorsionFingerprint": molecule_GetHashedTopologicalTorsionFingerprint,
    "molecule_GetHashedTopologicalTorsionFingerprintAsBitVect": molecule_GetHashedTopologicalTorsionFingerprintAsBitVect,
    "molecule_GetMACCSKeysFingerprint": molecule_GetMACCSKeysFingerprint,
    "molecule_LayeredFingerprint": molecule_LayeredFingerprint,
    "molecule_PatternFingerprint": molecule_PatternFingerprint,
    "molecule_RDKFingerprint": molecule_RDKFingerprint,
=======
        ConvertToNumpyArray(GetHashedTopologicalTorsionFingerprintAsBitVect(mol), a)
        return a


molecule_GetHashedAtomPairFingerprintAsBitVect = (
    GetHashedAtomPairFingerprintAsBitVect_Featurizer()
)
molecule_PatternFingerprint = PatternFingerprint_Featurizer()
molecule_GetMACCSKeysFingerprint = GetMACCSKeysFingerprint_Featurizer()
molecule_RDKFingerprint = RDKFingerprint_Featurizer()
molecule_GetHashedTopologicalTorsionFingerprint = (
    GetHashedTopologicalTorsionFingerprint_Featurizer()
)
molecule_LayeredFingerprint = LayeredFingerprint_Featurizer()
molecule_GetHashedAtomPairFingerprint = GetHashedAtomPairFingerprint_Featurizer()
molecule_GetHashedTopologicalTorsionFingerprintAsBitVect = (
    GetHashedTopologicalTorsionFingerprintAsBitVect_Featurizer()
)

_available_featurizer = {
    "molecule_GetHashedAtomPairFingerprintAsBitVect": molecule_GetHashedAtomPairFingerprintAsBitVect,
    "molecule_PatternFingerprint": molecule_PatternFingerprint,
    "molecule_GetMACCSKeysFingerprint": molecule_GetMACCSKeysFingerprint,
    "molecule_RDKFingerprint": molecule_RDKFingerprint,
    "molecule_GetHashedTopologicalTorsionFingerprint": molecule_GetHashedTopologicalTorsionFingerprint,
    "molecule_LayeredFingerprint": molecule_LayeredFingerprint,
    "molecule_GetHashedAtomPairFingerprint": molecule_GetHashedAtomPairFingerprint,
    "molecule_GetHashedTopologicalTorsionFingerprintAsBitVect": molecule_GetHashedTopologicalTorsionFingerprintAsBitVect,
>>>>>>> 3a4ef9b47d39261f7b23c48e5302154b6b288fc7
}


def main():
    from rdkit import Chem
<<<<<<< HEAD

    testmol = Chem.MolFromSmiles("c1ccccc1")
    for k, f in _available_featurizer.items():
        print(k, f(testmol))

=======

    testmol = Chem.MolFromSmiles("c1ccccc1")
    for k, f in _available_featurizer.items():
        print(k)
        f(testmol)

>>>>>>> 3a4ef9b47d39261f7b23c48e5302154b6b288fc7

if __name__ == "__main__":
    main()
