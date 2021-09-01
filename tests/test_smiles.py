import unittest

from rdkit import Chem
from molNet.mol.molecule import Molecule
from molNet.utils.smiles.modification import get_random_smiles


class SMILESTest(unittest.TestCase):
    def test_smiles_to_molecule(self):
        mol = Chem.MolFromSmiles("Cn1c(=O)c2c(ncn2C)n(C)c1=O")
        molecule = Molecule(mol)
        assert str(molecule) == "Cn1c(=O)c2c(ncn2C)n(C)c1=O"
        assert molecule.get_smiles() == "Cn1c(=O)c2c(ncn2C)n(C)c1=O"

    def test_random_smiles(self):
        smiles = "Cn1c(=O)c2c(ncn2C)n(C)c1=O"
        test_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
        rs = get_random_smiles(smiles, 100)
        for cs in rs:
            assert Chem.MolToSmiles(Chem.MolFromSmiles(cs)) == test_smiles
