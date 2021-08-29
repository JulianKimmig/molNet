import unittest

from rdkit import Chem


from molNet.mol.molecule import Molecule
from molNet.utils.identifier2smiles import name_to_smiles


class MolTest(unittest.TestCase):
    def test_mol_prop_hold(self):
        mol = Chem.MolFromSmiles("Cn1c(=O)c2c(ncn2C)n(C)c1=O")
        molecule = Molecule(mol)
        assert str(molecule) == "Cn1c(=O)c2c(ncn2C)n(C)c1=O"
        molecule.set_property("name", "caffein")
        assert molecule.get_property("name") == "caffein"
        mol = molecule.get_mol()
        molecule = Molecule(mol)
        assert str(molecule) == "caffein"

    def test_mol_from_name(self):
        from_name = "caffein"
        soll_string = "Cn1c(=O)c2c(ncn2C)n(C)c1=O"

        ns = name_to_smiles(from_name)
        mol = Chem.MolFromSmiles(list(ns.keys())[0])
        molecule = Molecule(mol)
        assert str(molecule) == "Cn1c(=O)c2c(ncn2C)n(C)c1=O"
        molecule.set_property("name", from_name)

        assert molecule.get_smiles() == soll_string

    def test_random_smiles(self):
        mol = Chem.MolFromSmiles("Cn1c(=O)c2c(ncn2C)n(C)c1=O")
        molecule = Molecule(mol)

    def test_representations(self):
        mol = Chem.MolFromSmiles("Cn1c(=O)c2c(ncn2C)n(C)c1=O")
        molecule = Molecule(mol)
        assert str(molecule) == "Cn1c(=O)c2c(ncn2C)n(C)c1=O"
        assert (
            "width='200px' height='200px' viewBox='0 0 200 200'>" in molecule.to_svg()
        )
        molecule.calc_position()

        print(molecule.as_dict())

        import pickle

        pm = molecule.get_mol()
        molecule = Molecule(pickle.loads(pickle.dumps(pm)))

        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".sdf", delete=False) as f:
            fn = f.name
            w = Chem.SDWriter(fn)
            pm = molecule.get_mol()
            w.write(pm)
            w = None

            with open(fn, "r") as inf:
                print(inf.read())
