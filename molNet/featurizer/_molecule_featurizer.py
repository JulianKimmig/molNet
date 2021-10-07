from .featurizer import Featurizer, FixedSizeFeaturizer
from molNet.utils.mol.properties import assert_conformers
import numpy as np
from molNet.utils.smiles import mol_from_smiles

from ._autogen_molecule_featurizer import *
testmol = mol_from_smiles("CCC")


class MoleculeFeaturizer(Featurizer):
    _LENGTH = None

    def pre_featurize(self, mol):
        mol = assert_conformers(mol)
        if self._add_prefeat:
            mol = self._add_prefeat(mol)
        return mol

    def __init__(self, *args, **kwargs):
        # if "length" not in kwargs:
        #    kwargs["length"] = self._LENGTH
        # if kwargs["length"] is None:
        #    kwargs["length"] = len(self.featurize(testmol))

        self._add_prefeat = kwargs.get("pre_featurize", None)
        kwargs["pre_featurize"] = None

        super().__init__(*args, **kwargs)


class SingleValueMoleculeFeaturizer(FixedSizeFeaturizer, MoleculeFeaturizer):
    LENGTH = 1
