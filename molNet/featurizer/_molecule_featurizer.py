import numpy as np
from rdkit import Chem
from rdkit.Chem import AddHs, RenumberAtoms, CanonicalRankAtoms, SanitizeMol
from rdkit.Chem.PropertyMol import PropertyMol
from rdkit.Chem.rdchem import Mol

from molNet.utils.mol.properties import assert_conformers
from molNet.utils.smiles import mol_from_smiles
from molNet.featurizer.featurizer import Featurizer, FixedSizeFeaturizer, StringFeaturizer
from molNet import MOLNET_LOGGER

testmol = mol_from_smiles("CCC")


def prepare_mol_for_featurization(mol, addHs=True, renumber=True, conformers=True, sanitize=True) -> Mol:

    if addHs:
        if mol.GetNumAtoms(onlyHeavy=False)!=mol.GetNumAtoms(onlyHeavy=True):
            mol = AddHs(mol,addCoords=True)

    if conformers:
        mol = assert_conformers(mol)

    _props = mol.GetPropsAsDict()
    if mol.HasProp("_Name"):
        _props["_Name"]=mol.GetProp("_Name")

    if renumber:
        mol = RenumberAtoms(
            mol, np.argsort(CanonicalRankAtoms(mol)).tolist()
        )

    if sanitize:
        SanitizeMol(mol)
    mol = PropertyMol(mol)
    for k,v in _props.items():
        mol.SetProp(k,v)
    mol.SetProp('_is_prepared', 1)
    return mol


def check_mol_is_prepared(mol):
    if not mol.HasProp('_is_prepared'):
        return False
    if mol.GetProp('_is_prepared') == "1":
        return True
    if mol.GetProp('_is_prepared') == 1:
        return True
    if mol.GetProp('_is_prepared') == "True":
        return True
    if mol.GetProp('_is_prepared') == True:
        return True
    return False


def mol_pre_featurize(featurizer,mol):
    mol=Mol(mol)#autocopy
    if not check_mol_is_prepared(mol):
        if not featurizer._unprepared_logged:
            MOLNET_LOGGER.warning("you tried to featurize a molecule without previous preparation. "
                                  "I will do this for you, but please try to implement this, "
                                  "otherwise you might end uo with differences in yout molecules and the featurized,"
                                  " since the preparation creates an copy of the molecule, "
                                  "adds hydrogens, conformerst etc."
                                  "")
            featurizer._unprepared_logged = True
        mol = prepare_mol_for_featurization(mol)
    return mol

class _MoleculeFeaturizer(Featurizer):
    def __init__(self, *args, **kwargs):
        self._unprepared_logged = False
        super().__init__(*args, **kwargs)
        self.prepend_prefeaturizer(mol_pre_featurize,"mol_pre_featurize")


class VarSizeMoleculeFeaturizer(_MoleculeFeaturizer, Featurizer):
    pass


MoleculeFeaturizer = VarSizeMoleculeFeaturizer


class FixedSizeMoleculeFeaturizer(_MoleculeFeaturizer, FixedSizeFeaturizer):
    pass


class SingleValueMoleculeFeaturizer(FixedSizeMoleculeFeaturizer):
    LENGTH = 1


class StringMoleculeFeaturizer(_MoleculeFeaturizer, StringFeaturizer):
    pass


class MoleculeHasSubstructureFeaturizer(SingleValueMoleculeFeaturizer):
    dtype: np.dtype = bool
    SMARTS: str = "#"

    def __init__(self, *args, smarts=None, **kwargs):
        super().__init__(*args, **kwargs)
        if smarts is None:
            smarts = self.SMARTS
        self._smarts = smarts

        self._pattern = Chem.MolFromSmarts(self._smarts)

    def featurize(self, mol):
        return mol.HasSubstructMatch(self._pattern)
