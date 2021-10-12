from rdkit import Chem

from .featurizer import Featurizer, FixedSizeFeaturizer
from molNet.utils.mol.properties import assert_conformers
import numpy as np
from molNet.utils.smiles import mol_from_smiles


testmol = mol_from_smiles("CCC")


class _MoleculeFeaturizer(Featurizer):

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


class VarSizeMoleculeFeaturizer(_MoleculeFeaturizer,Featurizer):
    pass

MoleculeFeaturizer = VarSizeMoleculeFeaturizer

class FixedSizeMoleculeFeaturizer(_MoleculeFeaturizer,FixedSizeFeaturizer):
    pass

class SingleValueMoleculeFeaturizer(FixedSizeMoleculeFeaturizer):
    LENGTH = 1



class MoleculeHasSubstructureFeaturizer(SingleValueMoleculeFeaturizer):
    dtype:np.dtype = bool
    SMARTS:str= "#"

    def __init__(self, *args,smarts=None, **kwargs):
        super().__init__(*args, **kwargs)
        if smarts is None:
            smarts = self.SMARTS
        self._smarts = smarts

        self._pattern = Chem.MolFromSmarts(self._smarts)

    def featurize(self,mol):
        return mol.HasSubstructMatch(self._pattern)
