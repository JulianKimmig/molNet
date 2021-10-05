from rdkit.Chem.rdmolops import GetMolFrags

from ._autogen_molecule_featurizer import *
from .featurizer import FeaturizerList


class ExtendMolnetFeaturizer(MoleculeFeaturizer):
    def featurize(self, mol):
        r = mol.molnet_features if hasattr(mol, "molnet_features") else []
        return np.array([r]).flatten()


extend_molnet_featurizer = ExtendMolnetFeaturizer(name="extend_molnet_featurizer")


class NumAtomsFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.int32

    def featurize_function(self, mol):
        return mol.GetNumAtoms()


molecule_num_atoms = NumAtomsFeaturizer()


class NumBondsFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.int32

    def featurize_function(self, mol):
        return mol.GetNumBonds()


molecule_num_bonds = NumBondsFeaturizer()


class NumFragmentsFeaturizer(SingleValueMoleculeFeaturizer):
    dtype = np.int32

    def featurize_function(self, mol):
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
        molecule_heavy_atom_count,
        molecule_crippen_descriptors,
        molecule_num_atoms,
        molecule_num_bonds,
        molecule_num_aliphatic_carbocycles,
        molecule_num_aliphatic_heterocycles,
        molecule_num_aliphatic_rings,
        molecule_num_aromatic_carbocycles,
        molecule_num_aromatic_heterocycles,
        molecule_num_aromatic_rings,
        # molecule_num_atom_stereo_centers,
        molecule_num_bridgehead_atoms,
        molecule_num_hba,
        molecule_num_hbd,
        molecule_num_heteroatoms,
        #    molecule_num_heterocycles,
        #    molecule_num_lipinski_hba,
        #    molecule_num_lipinski_hbd,
        molecule_num_rings,
        molecule_num_rotatable_bonds,
        #    molecule_num_saturated_carbocycles,
        #    molecule_num_saturated_heterocycles,
        molecule_num_saturated_rings,
        molecule_num_spiro_atoms,
        #    molecule_num_unspecified_atom_stereo_centers
    ],
    name="default_molecule_featurizer",
)
