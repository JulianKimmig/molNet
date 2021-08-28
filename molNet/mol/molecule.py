from typing import Union, List, Tuple, Any, Dict

import numpy as np
import rdkit
from rdkit.Chem import MolToSmiles, MolFromSmiles, MolFromInchi, rdmolfiles, rdmolops
from rdkit.Chem.PropertyMol import PropertyMol
from rdkit.Chem.rdchem import EditableMol

from molNet.utils.identifier2smiles import name_to_smiles

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


class SMILEError(Exception):
    pass


class MolGenerationError(Exception):
    pass


class Molecule(MolDataPropertyHolder):
    def __init__(self, mol, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._mol = mol

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

        self.smiles = MolToSmiles(mol)

    @property
    def mol(self):
        return self._mol

    @property
    def smiles(self):
        return self.get_property("smiles")

    @smiles.setter
    def smiles(self, smiles):
        to_s = MolToSmiles(self.mol)
        if smiles != to_s:
            conv_s = MolToSmiles(MolFromSmiles(smiles, sanitize=False))
            if not to_s == conv_s:
                raise SMILEError(
                    "smiles dont match {} and {} as {}".format(to_s, smiles, conv_s)
                )
        self.set_property("smiles", smiles, dtype=DATATYPES.STRING)

    def get_smiles(self, *args, **kwargs):
        return MolToSmiles(self.mol, *args, **kwargs)

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
        with_numbers=False,
        with_H=None,
        canonical_rank=True,
        with_properties=True,
    ) -> rdkit.Chem.Mol:
        """

        :rtype: rdkit.Chem.Mol
        """
        mol = EditableMol(self._mol)
        mol = mol.GetMol()

        if with_H:
            mol = rdkit.Chem.AddHs(mol)
        elif with_H is None:
            pass
        else:
            mol = rdkit.Chem.RemoveHs(mol)

        if canonical_rank:
            atom_order = rdmolfiles.CanonicalRankAtoms(mol)
            mol = rdmolops.RenumberAtoms(mol, atom_order)

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

    def __str__(self) -> str:
        _name: Union[str, None] = self.get_property("name")
        if _name is not None and len(_name) > 0:
            return _name
        return MolToSmiles(self._mol)


def molecule_from_name(name, *args, **kwargs):
    return Molecule.from_name(name, *args, **kwargs)


def molecule_from_smiles(smiles, *args, **kwargs):
    return Molecule.from_smiles(smiles, *args, **kwargs)


def molecule_from_inchi(inchi, *args, **kwargs):
    return Molecule.from_inchi(inchi, *args, **kwargs)
