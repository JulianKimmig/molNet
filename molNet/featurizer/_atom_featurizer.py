from .featurizer import Featurizer
import numpy as np
from rdkit.Chem import MolFromSmiles

testmol = MolFromSmiles("CCC")


class AtomFeaturizer(Featurizer):
    _LENGTH = None

    def __init__(self, **kwargs):
        if "length" not in kwargs:
            kwargs["length"] = self._LENGTH
            kwargs["length"] = self._LENGTH
        if kwargs["length"] is None:
            kwargs["length"] = len(self.featurize(testmol.GetAtomWithIdx(0)))

        if "pre_featurize" in kwargs:
            ipf = kwargs["pre_featurize"]

            def _pf(atom):
                return ipf(atom)

        else:

            def _pf(atom):
                return atom

        kwargs["pre_featurize"] = _pf
        super().__init__(**kwargs)


class SingleValueAtomFeaturizer(AtomFeaturizer):
    _LENGTH = 1

    def featurize(self, atom):
        return np.array([self.featurize_function(atom)], dtype=self.dtype)

    def featurize_function(self, atom):
        raise NotImplementedError()
