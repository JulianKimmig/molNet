import unittest

from rdkit import Chem

from molNet.mol.molecules import Molecule, molecule_from_name
from molNet.utils.identifier2smiles import name_to_smiles


class MolTest(unittest.TestCase):
    def test_mol_from_name(self):
        from_name="caffein"
        soll_string = "Cn1c(=O)c2c(ncn2C)n(C)c1=O"

        ns = name_to_smiles(from_name)
        mol = Chem.MolFromSmiles(list(ns.keys())[0])
        molecule = Molecule(mol,from_name)

        assert molecule.get_smiles() == soll_string
        assert molecule_from_name(from_name).get_smiles() == soll_string
        assert Molecule.from_name(from_name).get_smiles() == soll_string