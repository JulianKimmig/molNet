import unittest

import rdkit

from molNet.featurizer import atom_featurizer, molecule_featurizer


class MolFeaturizeTest(unittest.TestCase):
    def test_general_atom_feats(self):
        for featurizer in atom_featurizer.__all__:
            mol = rdkit.Chem.MolFromSmiles("CCCCC")
            f = getattr(atom_featurizer, featurizer)
            for atom in mol.GetAtoms():
                f(atom)

    def test_general_mol_feats(self):
        for featurizer in molecule_featurizer.__all__:
            mol = rdkit.Chem.MolFromSmiles("c1ccccc1CCl")
            f = getattr(molecule_featurizer, featurizer)
            f(mol)
