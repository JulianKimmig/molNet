import pickle
import unittest

import numpy as np
from rdkit.Chem import MolFromSmiles

from molNet.mol.molgraph import mol_graph_from_mol, mol_graph_from_smiles
from molNet.utils.mol.properties import assert_conformers


class MolGraphTest(unittest.TestCase):
    def test_gen_mol_graph_from_mol(self):
        smiles = "Cn1c(=O)c2c(ncn2C)n(C)c1=O"
        mol = MolFromSmiles(smiles)
        assert_conformers(mol)
        assert len(mol_graph_from_mol(mol)) == 24

    def test_gen_mol_graph_from_smiles(self):
        smiles = "Cn1c(=O)c2c(ncn2C)n(C)c1=O"
        assert len(mol_graph_from_smiles(smiles)) == 24

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

    def test_freezing(self):
        mol = MolFromSmiles("Cn1c(=O)c2c(ncn2C)n(C)c1=O")
        mg = mol_graph_from_mol(mol)
        from molNet.featurizer.atom_featurizer import atom_mass
        from molNet.featurizer.molecule_featurizer import molecule_mol_wt

        mg.featurize_mol(molecule_mol_wt, "mwf")
        mg.featurize_atoms(atom_mass, "mwf")

        fmg = mg.freeze()
        fmgd = fmg.as_arrays()
        mgd = mg.as_arrays()
        assert fmgd["size"] == mgd["size"]
        assert np.allclose(fmgd["eges"], mgd["eges"])
        for k, v in mgd["node_features"].items():
            if isinstance(v, np.ndarray):
                assert np.allclose(fmgd["node_features"][k], v)
        for k, v in mgd["graph_features"].items():
            if isinstance(v, np.ndarray):
                assert np.allclose(fmgd["graph_features"][k], v)

    def test_molgraphs_equal(self):
        from molNet.mol.molgraph import (
            assert_molgraphs_data_equal,
            MolgraphEqualsException,
        )

        mol = MolFromSmiles("Cn1c(=O)c2c(ncn2C)n(C)c1=O")
        mg1 = mol_graph_from_mol(mol)
        mg2 = mol_graph_from_mol(mol)
        from molNet.featurizer.atom_featurizer import atom_mass
        from molNet.featurizer.molecule_featurizer import molecule_mol_wt

        mg1.featurize_mol(molecule_mol_wt, "mwf")
        mg2.featurize_mol(molecule_mol_wt, "mwf")

        assert_molgraphs_data_equal(mg1, mg2)

        mg2.featurize_atoms(atom_mass, "mwf")

        with self.assertRaises(MolgraphEqualsException) as context:
            assert_molgraphs_data_equal(mg1, mg2)
        assert str(context.exception) == "feature missmatch('node_features,mwf')", str(
            context.exception
        )

        mg3 = mol_graph_from_mol(mg1.mol)
        mg3.featurize_mol(molecule_mol_wt, "mwf")
        assert_molgraphs_data_equal(mg1, mg3)

    def test_pickling(self):
        import tempfile
        import os

        mol = MolFromSmiles("Cn1c(=O)c2c(ncn2C)n(C)c1=O")
        mg = mol_graph_from_mol(mol)
        from molNet.featurizer.atom_featurizer import atom_mass
        from molNet.featurizer.molecule_featurizer import molecule_mol_wt

        mg.featurize_mol(molecule_mol_wt, "mwf")
        mg.featurize_atoms(atom_mass, "mwf")

        fmg = mg.freeze()

        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            with open(tmp_file.name, "wb") as sf:
                pickle.dump(mg, sf)
            tmp_filename = tmp_file.name

        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            with open(tmp_file.name, "wb") as sf:
                pickle.dump(fmg, sf)
            ftmp_filename = tmp_file.name

        assert os.path.getsize(ftmp_filename) == 2132, os.path.getsize(ftmp_filename)
        assert os.path.getsize(tmp_filename) == 2573, os.path.getsize(tmp_filename)

        with open(tmp_filename, "rb") as sf:
            nmg = pickle.load(sf)

        with open(ftmp_filename, "rb") as sf:
            nfmg = pickle.load(sf)

        os.remove(tmp_filename)
        os.remove(ftmp_filename)

        nfmgd = nfmg.as_arrays()
        fmgd = nfmg.as_arrays()
        assert fmgd["size"] == nfmgd["size"]
        assert np.allclose(fmgd["eges"], nfmgd["eges"])
        for k, v in fmgd["node_features"].items():
            if isinstance(v, np.ndarray):
                assert np.allclose(nfmgd["node_features"][k], v)
        for k, v in fmgd["graph_features"].items():
            if isinstance(v, np.ndarray):
                assert np.allclose(nfmgd["graph_features"][k], v)
