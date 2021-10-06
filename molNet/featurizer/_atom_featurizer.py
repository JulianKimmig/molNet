from .featurizer import Featurizer, FixedSizeFeaturizer
import numpy as np
from molNet.utils.smiles import mol_from_smiles

testmol = mol_from_smiles("CCC")


class AtomFeaturizer(Featurizer):
    pass


class SingleValueAtomFeaturizer(FixedSizeFeaturizer, AtomFeaturizer):
    LENGTH = 1
