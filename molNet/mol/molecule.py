from typing import Union, List, Tuple, Any, Dict

import numpy as np
import rdkit
from rdkit.Chem import MolToSmiles, MolFromSmiles, MolFromInchi, rdmolfiles, rdmolops
from rdkit.Chem.PropertyMol import PropertyMol
from rdkit.Chem.rdchem import Mol

from molNet import MolGenerationError
from molNet.utils.identifier2smiles import name_to_smiles
from molNet.utils.mol.draw import mol_to_svg
from molNet.utils.mol.properties import assert_conformers

DATATYPES_MAP: Dict[str, Tuple[type]] = {
    "FLOAT": (np.floating, float),
    "INT": (np.integer, int),
    "STRING": (str, np.str_),
    "NONE": (type(None)),
    "BOOL": (bool, np.bool_),
}
DATATYPES = type("", (), {})()
DATATYPES.STRING = "STRING"
DATATYPES.NONE = "NONE"
DATATYPES.INT = "INT"
DATATYPES.FLOAT = "FLOAT"
DATATYPES.BOOL = "BOOL"


class MolDataPropertyHolder:
    @staticmethod
    def get_dtype(obj):
        if isinstance(obj, np.ndarray):
            ta = np.array([1.1]).astype(obj.dtype)
            for key, val in DATATYPES_MAP.items():
                if isinstance(ta[0], val):
                    return key
        if isinstance(obj, DATATYPES_MAP[DATATYPES.STRING]):
            return DATATYPES.STRING
        try:
            dtypes = [MolDataPropertyHolder.get_dtype(sobj) for sobj in obj]
            if len(set(dtypes)) == 1:
                return dtypes[0]
            return DATATYPES.NONE
        except:
            pass

        for key, val in DATATYPES_MAP.items():
            if isinstance(obj, val):
                return key
        return DATATYPES.NONE

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._properties = {k: {} for k in DATATYPES_MAP.keys()}
        self._names = []
        self._dtypes = []

    def set_property(self, name, value, dtype=None):
        if dtype is None:
            dtype = MolDataPropertyHolder.get_dtype(value)

        if dtype not in DATATYPES_MAP.keys():
            raise TypeError("unknown type '{}'".format(dtype))

        if name in self._names:
            or_index = self._names.index(name)
            or_key = self._dtypes[or_index]
            if or_key != dtype:
                del self._properties[or_key][name]
                self._dtypes[or_index] = dtype
        else:
            self._names.append(name)
            self._dtypes.append(dtype)
        self._properties[dtype][name] = value

    def get_property(self, name, with_dtype=False):
        if name not in self._names:
            return None
        index = self._names.index(name)
        dtype = self._dtypes[index]
        if with_dtype:
            return self._properties[dtype][name], dtype
        return self._properties[dtype][name]

    def has_property(self, name):
        return name in self._properties

    def get_property_names_with_type(self) -> List[Tuple[str, Any]]:
        return list(zip(self._names.copy(), self._dtypes.copy()))

    def get_property_names(self) -> List[str]:
        return self._names.copy()

    def get_properties(self) -> Dict[str, Any]:
        return {p: self.get_property(p) for p in self.get_property_names()}


def perpare_mol_for_molecule(mol: Mol) -> Mol:
    mol = rdkit.Chem.AddHs(mol)
    mol = rdmolops.RenumberAtoms(
        mol, np.argsort(rdmolfiles.CanonicalRankAtoms(mol)).tolist()
    )
    return mol


class Molecule(MolDataPropertyHolder):
    def __init__(self, mol: Mol, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._mol: Mol = perpare_mol_for_molecule(mol)

        # restore properties
        _propnames: List[str] = mol.GetPropNames()
        for p in _propnames:
            if p.endswith("__type"):
                continue
            v = mol.GetProp(p)
            t = DATATYPES.STRING
            if p + "__type" in _propnames:
                t = mol.GetProp(p + "__type")
                v = DATATYPES_MAP[t][0](v)

            if p.startswith("molNet_"):
                p = p.replace("molNet_", "", 1)
            self.set_property(p, v, t)

        # self.smiles = MolToSmiles(self._mol)

    @property
    def mol(self):
        return self._mol

    # @property
    # def smiles(self):
    #    return self.get_property("smiles")

    # @smiles.setter
    # def smiles(self, smiles):
    #    to_s = MolToSmiles(self.mol)
    #    if smiles != to_s:
    #        conv_s = MolToSmiles(MolFromSmiles(smiles, sanitize=False))
    #        if not to_s == conv_s:
    #            raise SMILEError(
    #                "smiles dont match {} and {} as {}".format(to_s, smiles, conv_s)
    #            )
    #    self.set_property("smiles", smiles, dtype=DATATYPES.STRING)

    def get_smiles(self, *args, **kwargs):
        return MolToSmiles(rdkit.Chem.RemoveHs(self.mol), *args, **kwargs)

    @classmethod
    def from_smiles(cls, smiles: str, *args, **kwargs):
        m = MolFromSmiles(smiles)
        if m is None:
            raise MolGenerationError(
                "cannot convert smiles '{}' to molecule".format(smiles)
            )
        m = cls(m, *args, **kwargs)
        return m

    @classmethod
    def from_inchi(cls, inchi: str, *args, **kwargs):
        m = MolFromInchi(inchi)
        if m is None:
            raise MolGenerationError(
                "cannot convert inchi '{}' to molecule".format(inchi)
            )
        m = cls(m, *args, **kwargs)
        return m

    @classmethod
    def from_name(cls, name: str, *args, **kwargs):
        ns = name_to_smiles(name)
        if len(ns) == 0:
            MolGenerationError("no SMILE from name")

        molecule = cls.from_smiles(smiles=list(ns.keys())[0], *args, **kwargs)
        molecule.set_property("name", name, DATATYPES.STRING)
        return molecule

    def get_mol(
        self,
        with_numbers: bool = False,
        with_H: bool = None,
        canonical_rank: bool = True,
        with_properties: bool = True,
    ) -> Mol:
        mol = PropertyMol(self._mol)  # TODO find better copy method?

        if with_H:
            nmol = rdkit.Chem.AddHs(mol)
            if mol.GetNumAtoms() != nmol.GetNumAtoms():
                mol = nmol
        elif with_H is None:
            pass
        else:
            nmol = rdkit.Chem.RemoveHs(mol)
            if mol.GetNumAtoms() != nmol.GetNumAtoms():
                mol = nmol

        if canonical_rank:
            mol = rdmolops.RenumberAtoms(
                mol, np.argsort(rdmolfiles.CanonicalRankAtoms(mol)).tolist()
            )

        if with_numbers:
            atoms = mol.GetNumAtoms()
            for idx in range(atoms):
                mol.GetAtomWithIdx(idx).SetProp(
                    "molAtomMapNumber", str(mol.GetAtomWithIdx(idx).GetIdx())
                )
        if with_properties:
            mol = PropertyMol(mol)
            for p, t in self.get_property_names_with_type():
                mol.SetProp("molNet_" + p, str(self.get_property(p)))
                # if t != DATATYPES.STRING:
                mol.SetProp("molNet_" + p + "__type", t)
        return mol

    # representations

    def to_svg(self, size=(200, 200), svg_data=None, mol_data=None):
        if mol_data is None:
            mol_data = {}
        return mol_to_svg(mol=self.get_mol(**mol_data), size=size, svg_data=svg_data)

    def _repr_svg_(self):
        return self.to_svg()

    def __str__(self) -> str:
        _name: Union[str, None] = self.get_property("name")
        if _name is not None and len(_name) > 0:
            return _name
        return self.get_smiles()

    def as_dict(self):
        return {
            "properties": self._properties,
            "mol_properties": self.get_mol(with_properties=False).GetPropsAsDict(
                includeComputed=True
            ),
        }

    def calc_position(self, norm=True):
        mol = assert_conformers(self.mol)

        c = mol.GetConformers()[0]
        pos = c.GetPositions()
        pos = pos[:, :2]
        pos = {i: pos[i] for i in range(pos.shape[0])}

        if norm:
            pos_list = np.zeros((len(pos), 2))
            for i in range(pos_list.shape[0]):
                pos_list[i] = pos[i]
            pos_list[:, 0] -= pos_list[:, 0].min()
            pos_list[:, 1] -= pos_list[:, 1].min()
            pos_list /= pos_list.max()

            pos = {i: pos_list[i] for i in range(pos_list.shape[0])}
        return pos


def molecule_from_name(name, *args, **kwargs):
    return Molecule.from_name(name, *args, **kwargs)


def molecule_from_smiles(smiles, *args, **kwargs):
    return Molecule.from_smiles(smiles, *args, **kwargs)


def molecule_from_inchi(inchi, *args, **kwargs):
    return Molecule.from_inchi(inchi, *args, **kwargs)
