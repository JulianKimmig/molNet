import unittest

from rdkit.Chem import MolFromSmiles

from molNet.mol.molgraph import mol_graph_from_mol
from molNet.utils.mol.properties import assert_confomers


class MolGraphTest(unittest.TestCase):
    def test_gen_mol_graph_from_mol(self):
        mol = MolFromSmiles("Cn1c(=O)c2c(ncn2C)n(C)c1=O")
        assert_confomers(mol)
        assert len(mol_graph_from_mol(mol)) == 24

    def test_featurize_molecule(self):
        mol = MolFromSmiles("Cn1c(=O)c2c(ncn2C)n(C)c1=O")
        mg = mol_graph_from_mol(mol)

        from molNet.featurizer.molecule_featurizer import molecule_mol_wt

        mg.featurize_mol(molecule_mol_wt, "mwf")
        assert mg.get_property("mwf") == 194.194

    def test_featurize_atoms(self):
        mol = MolFromSmiles("Cn1c(=O)c2c(ncn2C)n(C)c1=O")
        mg = mol_graph_from_mol(mol)

        from molNet.featurizer.atom_featurizer import atom_mass

        mg.featurize_atoms(atom_mass, "mwf")

        from molNet.featurizer.molecule_featurizer import molecule_mol_wt

        mg.featurize_mol(molecule_mol_wt, "mwf")

        print(mg.nodes(data=True))
        print(mg.get_atom_features_dict())

        assert mg.get_property("mwf") == mg.get_atom_feature("mwf").sum()

    def test_graph_as_array(self):
        mol = MolFromSmiles("Cn1c(=O)c2c(ncn2C)n(C)c1=O")
        mg = mol_graph_from_mol(mol)
        from molNet.featurizer.atom_featurizer import atom_mass
        from molNet.featurizer.molecule_featurizer import molecule_mol_wt

        mg.featurize_mol(molecule_mol_wt, "mwf")
        mg.featurize_atoms(atom_mass, "mwf")

        md_data = mg.as_arrays()
        assert md_data["size"] == 24
        assert len(md_data["eges"]) == 25
        assert md_data["eges"].size == 50
        assert md_data["node_features"]["mwf"].sum() == md_data["graph_features"]["mwf"]
