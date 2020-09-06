import unittest

from rdkit import Chem

from molNet.mol.molecules import Molecule
from molNet.utils.identifier2smiles import name_to_smiles


class MolTest(unittest.TestCase):
    def test_mol_from_name(self):
        ns = name_to_smiles("caffein")
        mol = Chem.MolFromSmiles(list(ns.keys())[0])
        molecule = Molecule(mol,"caffein")
        assert molecule.get_smiles() == "Cn1c(=O)c2c(ncn2C)n(C)c1=O"