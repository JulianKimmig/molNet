import unittest

import numpy as np
from rdkit.Chem import MolFromSmiles
import torch_geometric

from molNet.featurizer._autogen_molecule_featurizer import (
    molecule_mol_wt,
    molecule_get_formal_charge,
)
from molNet.featurizer.atom_featurizer import (
    atom_partial_charge,
    atom_symbol_one_hot,
    atom_degree,
)
from molNet.mol.molgraph import mol_graph_from_mol
from molNet.nn.graph.torch_geometric import (
    molgraph_to_graph_input,
    assert_graph_input_data_equal,
    GraphInputEqualsException,
    assert_graph_input_shape_equal,
)


class PytorchGeometricTest(unittest.TestCase):
    def test_input_from_molgraph(self):
        mol = MolFromSmiles("Cn1c(=O)c2c(ncn2C)n(C)c1=O")
        mg = mol_graph_from_mol(mol)
        mg.featurize_mol(molecule_mol_wt)
        mg.featurize_mol(molecule_get_formal_charge, as_y=True)
        mg.featurize_atoms(atom_symbol_one_hot)
        mg.featurize_atoms(atom_degree)
        mg.featurize_atoms(atom_partial_charge, as_y=True)

        data = molgraph_to_graph_input(mg)

        for b in torch_geometric.data.DataLoader([data, data, data], batch_size=64):
            assert np.allclose(
                b.batch.detach().numpy(),
                np.concatenate([np.zeros(24), np.ones(24), np.ones(24) * 2]),
            )
            assert np.allclose(
                b.y_graph_features.detach().numpy().flatten(), np.zeros(3)
            )
            assert np.allclose(
                b.x_graph_features.detach().numpy().flatten(), np.ones(3) * 194.194
            )

    def test_input_equal(self):

        mol = MolFromSmiles("Cn1c(=O)c2c(ncn2C)n(C)c1=O")
        mg1 = mol_graph_from_mol(mol)
        mg2 = mol_graph_from_mol(mol)
        from molNet.featurizer.atom_featurizer import atom_mass
        from molNet.featurizer.molecule_featurizer import molecule_mol_wt

        mg1.featurize_mol(molecule_mol_wt, "mwf")
        mg2.featurize_mol(molecule_mol_wt, "mwf")

        gip1 = molgraph_to_graph_input(mg1)
        gip2 = molgraph_to_graph_input(mg2)

        assert_graph_input_data_equal(gip1, gip2)

        mg2.featurize_atoms(atom_mass, "mwf")
        gip2 = molgraph_to_graph_input(mg2)

        with self.assertRaises(GraphInputEqualsException) as context:
            assert_graph_input_data_equal(gip1, gip2)
        assert str(context.exception) == "feature shape missmatch('x')", str(
            context.exception
        )

        mg3 = mol_graph_from_mol(mg1.mol)
        mg3.featurize_mol(molecule_mol_wt, "mwf")
        gip3 = molgraph_to_graph_input(mg3)
        assert_graph_input_data_equal(gip1, gip3)

        mol2 = MolFromSmiles("c1ccccc1")
        mg4 = mol_graph_from_mol(mol2)
        mg4.featurize_mol(molecule_mol_wt, "mwf")

        gip4 = molgraph_to_graph_input(mg4)
        assert_graph_input_shape_equal(gip1, gip4)
