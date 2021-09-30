import logging
import sys
import unittest

from rdkit import Chem

from molNet import MOLNET_LOGGER
from molNet.mol.molecule import Molecule
from molNet.utils.smiles.generator import (
    generate_random_carbon_lattice,
    generate_n_random_carbon_lattice,
)
from molNet.utils.smiles.modification import (
    get_random_smiles,
    multiple_mol_from_smiles,
    parallel_mol_from_smiles,
)

MOLNET_LOGGER.setLevel("DEBUG")


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

    def test_multicore_smiles(self):
        MOLNET_LOGGER.setLevel("DEBUG")
        smiles = generate_n_random_carbon_lattice(1000, progess_bar=True)
        ms = parallel_mol_from_smiles(smiles)
        mm = multiple_mol_from_smiles(smiles)
        for i, msi in enumerate(ms):
            assert Chem.MolToSmiles(msi) == Chem.MolToSmiles(mm[i])


logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
