from rdkit.Chem.rdmolops import GetMolFrags

from ._autogen_molecule_featurizer_list import *
from ._autogen_molecule_featurizer_numeric import *
from ._autogen_molecule_featurizer_numpy_arrays import *
from ._autogen_molecule_featurizer_rdkit_vec import *
from .featurizer import FeaturizerList


class ExtendMolnetFeaturizer(MoleculeFeaturizer):
    def featurize(self, mol):
        r = mol.molnet_features if hasattr(mol, "molnet_features") else []
        return np.array([r]).flatten()


extend_molnet_featurizer = ExtendMolnetFeaturizer(name="extend_molnet_featurizer")


class NumAtomsFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.int32

    def featurize(self, mol):
        return mol.GetNumAtoms()


molecule_num_atoms = NumAtomsFeaturizer()


class NumBondsFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.int32

    def featurize(self, mol):
        return mol.GetNumBonds()


molecule_num_bonds = NumBondsFeaturizer()


class NumFragmentsFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.int32

    def featurize(self, mol):
        return len(GetMolFrags(mol))


molecule_num_fragments = NumFragmentsFeaturizer()


from ._autogen_molecule_featurizer import _available_featurizer as _agaf

_available_featurizer = {
    **_agaf,
    "molecule_num_atoms": molecule_num_atoms,
    "molecule_num_bonds": molecule_num_bonds,
}

default_molecule_featurizer = FeaturizerList(
    [
        
    ],
    name="default_molecule_featurizer",
)
