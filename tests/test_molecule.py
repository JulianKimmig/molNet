import unittest


class MolTest(unittest.TestCase):
    def test_mol_from_name(self):
        from rdkit import Chem
        from molNet.mol.molecule import Molecule, molecule_from_name
        from molNet.utils.identifier2smiles import name_to_smiles

        from_name = "caffein"
        soll_string = "Cn1c(=O)c2c(ncn2C)n(C)c1=O"

        ns = name_to_smiles(from_name)
        mol = Chem.MolFromSmiles(list(ns.keys())[0])
        molecule = Molecule(mol)
        assert str(molecule) == "Cn1c(=O)c2c(ncn2C)n(C)c1=O"
        molecule.set_property("name", from_name)

        assert molecule.get_smiles() == soll_string
        assert molecule.get_property("name") == "caffein"
        assert molecule_from_name(from_name).get_smiles() == soll_string
        assert Molecule.from_name(from_name).get_smiles() == soll_string
        mol = molecule.get_mol()
        molecule = Molecule(mol)
        assert str(molecule) == "caffein"
