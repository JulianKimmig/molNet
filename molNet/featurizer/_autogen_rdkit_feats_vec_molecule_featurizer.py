import numpy as np
from molNet.featurizer._molecule_featurizer import (
    FixedSizeMoleculeFeaturizer,
    VarSizeMoleculeFeaturizer,
)
from rdkit.DataStructs import ConvertToNumpyArray
from rdkit.Chem.AllChem import (
    GetHashedAtomPairFingerprint,
    GetHashedTopologicalTorsionFingerprint,
    GetMACCSKeysFingerprint,
    GetHashedAtomPairFingerprintAsBitVect,
    GetHashedTopologicalTorsionFingerprintAsBitVect,
    LayeredFingerprint,
    RDKFingerprint,
    PatternFingerprint,
)
from rdkit.Chem import RDKFingerprint, PatternFingerprint, LayeredFingerprint


class Molecule_AllChem_HashedAtomPairFingerprintAsBitVect_Featurizer(
    FixedSizeMoleculeFeaturizer
):
    # _rdfunc=rdkit.Chem.AllChem.GetHashedAtomPairFingerprintAsBitVect
    dtype = bool
    LENGTH = 2048

    def featurize(self, mol):
        a = np.zeros(len(self), dtype=self.dtype)
        ConvertToNumpyArray(GetHashedAtomPairFingerprintAsBitVect(mol), a)
        return a


class Molecule_AllChem_HashedAtomPairFingerprint_Featurizer(
    FixedSizeMoleculeFeaturizer
):
    # _rdfunc=rdkit.Chem.AllChem.GetHashedAtomPairFingerprint
    dtype = np.int32
    LENGTH = 2048

    def featurize(self, mol):
        a = np.zeros(len(self), dtype=self.dtype)
        ConvertToNumpyArray(GetHashedAtomPairFingerprint(mol), a)
        return a


class Molecule_AllChem_HashedTopologicalTorsionFingerprintAsBitVect_Featurizer(
    FixedSizeMoleculeFeaturizer
):
    # _rdfunc=rdkit.Chem.AllChem.GetHashedTopologicalTorsionFingerprintAsBitVect
    dtype = bool
    LENGTH = 2048

    def featurize(self, mol):
        a = np.zeros(len(self), dtype=self.dtype)
        ConvertToNumpyArray(GetHashedTopologicalTorsionFingerprintAsBitVect(mol), a)
        return a


class Molecule_AllChem_HashedTopologicalTorsionFingerprint_Featurizer(
    FixedSizeMoleculeFeaturizer
):
    # _rdfunc=rdkit.Chem.AllChem.GetHashedTopologicalTorsionFingerprint
    dtype = np.int64
    LENGTH = 2048

    def featurize(self, mol):
        a = np.zeros(len(self), dtype=self.dtype)
        ConvertToNumpyArray(GetHashedTopologicalTorsionFingerprint(mol), a)
        return a


class Molecule_AllChem_LayeredFingerprint_Featurizer(FixedSizeMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.AllChem.LayeredFingerprint
    dtype = bool
    LENGTH = 2048

    def featurize(self, mol):
        a = np.zeros(len(self), dtype=self.dtype)
        ConvertToNumpyArray(LayeredFingerprint(mol), a)
        return a


class Molecule_AllChem_MACCSKeysFingerprint_Featurizer(FixedSizeMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.AllChem.GetMACCSKeysFingerprint
    dtype = bool
    LENGTH = 167

    def featurize(self, mol):
        a = np.zeros(len(self), dtype=self.dtype)
        ConvertToNumpyArray(GetMACCSKeysFingerprint(mol), a)
        return a


class Molecule_AllChem_PatternFingerprint_Featurizer(FixedSizeMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.AllChem.PatternFingerprint
    dtype = bool
    LENGTH = 2048

    def featurize(self, mol):
        a = np.zeros(len(self), dtype=self.dtype)
        ConvertToNumpyArray(PatternFingerprint(mol), a)
        return a


class Molecule_AllChem_RDKFingerprint_Featurizer(FixedSizeMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.AllChem.RDKFingerprint
    dtype = bool
    LENGTH = 2048

    def featurize(self, mol):
        a = np.zeros(len(self), dtype=self.dtype)
        ConvertToNumpyArray(RDKFingerprint(mol), a)
        return a


class Molecule_Chem_LayeredFingerprint_Featurizer(FixedSizeMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.LayeredFingerprint
    dtype = bool
    LENGTH = 2048

    def featurize(self, mol):
        a = np.zeros(len(self), dtype=self.dtype)
        ConvertToNumpyArray(LayeredFingerprint(mol), a)
        return a


class Molecule_Chem_PatternFingerprint_Featurizer(FixedSizeMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.PatternFingerprint
    dtype = bool
    LENGTH = 2048

    def featurize(self, mol):
        a = np.zeros(len(self), dtype=self.dtype)
        ConvertToNumpyArray(PatternFingerprint(mol), a)
        return a


class Molecule_Chem_RDKFingerprint_Featurizer(FixedSizeMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.RDKFingerprint
    dtype = bool
    LENGTH = 2048

    def featurize(self, mol):
        a = np.zeros(len(self), dtype=self.dtype)
        ConvertToNumpyArray(RDKFingerprint(mol), a)
        return a


molecule_AllChem_HashedAtomPairFingerprintAsBitVect_featurizer = (
    Molecule_AllChem_HashedAtomPairFingerprintAsBitVect_Featurizer()
)
molecule_AllChem_HashedAtomPairFingerprint_featurizer = (
    Molecule_AllChem_HashedAtomPairFingerprint_Featurizer()
)
molecule_AllChem_HashedTopologicalTorsionFingerprintAsBitVect_featurizer = (
    Molecule_AllChem_HashedTopologicalTorsionFingerprintAsBitVect_Featurizer()
)
molecule_AllChem_HashedTopologicalTorsionFingerprint_featurizer = (
    Molecule_AllChem_HashedTopologicalTorsionFingerprint_Featurizer()
)
molecule_AllChem_LayeredFingerprint_featurizer = (
    Molecule_AllChem_LayeredFingerprint_Featurizer()
)
molecule_AllChem_MACCSKeysFingerprint_featurizer = (
    Molecule_AllChem_MACCSKeysFingerprint_Featurizer()
)
molecule_AllChem_PatternFingerprint_featurizer = (
    Molecule_AllChem_PatternFingerprint_Featurizer()
)
molecule_AllChem_RDKFingerprint_featurizer = (
    Molecule_AllChem_RDKFingerprint_Featurizer()
)
molecule_Chem_LayeredFingerprint_featurizer = (
    Molecule_Chem_LayeredFingerprint_Featurizer()
)
molecule_Chem_PatternFingerprint_featurizer = (
    Molecule_Chem_PatternFingerprint_Featurizer()
)
molecule_Chem_RDKFingerprint_featurizer = Molecule_Chem_RDKFingerprint_Featurizer()
_available_featurizer = {
    "molecule_AllChem_HashedAtomPairFingerprintAsBitVect_featurizer": molecule_AllChem_HashedAtomPairFingerprintAsBitVect_featurizer,
    "molecule_AllChem_HashedAtomPairFingerprint_featurizer": molecule_AllChem_HashedAtomPairFingerprint_featurizer,
    "molecule_AllChem_HashedTopologicalTorsionFingerprintAsBitVect_featurizer": molecule_AllChem_HashedTopologicalTorsionFingerprintAsBitVect_featurizer,
    "molecule_AllChem_HashedTopologicalTorsionFingerprint_featurizer": molecule_AllChem_HashedTopologicalTorsionFingerprint_featurizer,
    "molecule_AllChem_LayeredFingerprint_featurizer": molecule_AllChem_LayeredFingerprint_featurizer,
    "molecule_AllChem_MACCSKeysFingerprint_featurizer": molecule_AllChem_MACCSKeysFingerprint_featurizer,
    "molecule_AllChem_PatternFingerprint_featurizer": molecule_AllChem_PatternFingerprint_featurizer,
    "molecule_AllChem_RDKFingerprint_featurizer": molecule_AllChem_RDKFingerprint_featurizer,
    "molecule_Chem_LayeredFingerprint_featurizer": molecule_Chem_LayeredFingerprint_featurizer,
    "molecule_Chem_PatternFingerprint_featurizer": molecule_Chem_PatternFingerprint_featurizer,
    "molecule_Chem_RDKFingerprint_featurizer": molecule_Chem_RDKFingerprint_featurizer,
}
__all__ = [
    "Molecule_AllChem_HashedAtomPairFingerprintAsBitVect_Featurizer",
    "molecule_AllChem_HashedAtomPairFingerprintAsBitVect_featurizer",
    "Molecule_AllChem_HashedAtomPairFingerprint_Featurizer",
    "molecule_AllChem_HashedAtomPairFingerprint_featurizer",
    "Molecule_AllChem_HashedTopologicalTorsionFingerprintAsBitVect_Featurizer",
    "molecule_AllChem_HashedTopologicalTorsionFingerprintAsBitVect_featurizer",
    "Molecule_AllChem_HashedTopologicalTorsionFingerprint_Featurizer",
    "molecule_AllChem_HashedTopologicalTorsionFingerprint_featurizer",
    "Molecule_AllChem_LayeredFingerprint_Featurizer",
    "molecule_AllChem_LayeredFingerprint_featurizer",
    "Molecule_AllChem_MACCSKeysFingerprint_Featurizer",
    "molecule_AllChem_MACCSKeysFingerprint_featurizer",
    "Molecule_AllChem_PatternFingerprint_Featurizer",
    "molecule_AllChem_PatternFingerprint_featurizer",
    "Molecule_AllChem_RDKFingerprint_Featurizer",
    "molecule_AllChem_RDKFingerprint_featurizer",
    "Molecule_Chem_LayeredFingerprint_Featurizer",
    "molecule_Chem_LayeredFingerprint_featurizer",
    "Molecule_Chem_PatternFingerprint_Featurizer",
    "molecule_Chem_PatternFingerprint_featurizer",
    "Molecule_Chem_RDKFingerprint_Featurizer",
    "molecule_Chem_RDKFingerprint_featurizer",
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
