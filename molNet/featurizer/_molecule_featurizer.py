from .featurizer import Featurizer
from molNet.utils.mol.properties import assert_confomers
import numpy as np
from rdkit.Chem import MolFromSmiles

testmol = MolFromSmiles("CCC")


class MoleculeFeaturizer(Featurizer):
    _LENGTH = None

    def __init__(self, **kwargs):
        if "length" not in kwargs:
            kwargs["length"] = self._LENGTH
        if kwargs["length"] is None:
            kwargs["length"] = len(self.featurize(testmol))

        if "pre_featurize" in kwargs:
            ipf = kwargs["pre_featurize"]

            def _pf(mol):
                assert_confomers(mol)
                return ipf(mol)

        else:

            def _pf(mol):
                assert_confomers(mol)
                return mol

        kwargs["pre_featurize"] = _pf
        super().__init__(**kwargs)

    def featurize(self, mol):
        return np.array(self.featurize_function(mol), dtype=self.dtype)


class SingleValueMoleculeFeaturizer(MoleculeFeaturizer):
    _LENGTH = 1

    def featurize(self, mol):
        return np.array([self.featurize_function(mol)], dtype=self.dtype)
