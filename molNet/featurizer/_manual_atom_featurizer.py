import numpy as np
from rdkit.Chem.rdPartialCharges import ComputeGasteigerCharges
from rdkit.Chem.rdchem import Mol

from molNet.featurizer._atom_featurizer import FixedSizeAtomFeaturizer, OneHotAtomFeaturizer, SingleValueAtomFeaturizer
from molNet.featurizer._autogen_rdkit_atomtype_atom_featurizer import Atom_AllSymbolOneHot_Featurizer
from molNet.utils.mol import ATOMIC_SYMBOL_NUMBERS


class Atom_ConnectedAtoms_Featurizer(FixedSizeAtomFeaturizer):
    dtype = np.int32
    LENGTH = max(ATOMIC_SYMBOL_NUMBERS.values()) + 1  # +1 since 0  is possible
    atoms = list(ATOMIC_SYMBOL_NUMBERS.keys())

    def featurize(self, atom):
        connected_atom_types = np.zeros(self.LENGTH)
        for b in atom.GetBonds():
            connected_atom_types[b.GetOtherAtom(atom).GetAtomicNum()] += 1
        return connected_atom_types


atom_ConnectedAtoms_featurizer = Atom_ConnectedAtoms_Featurizer()


def atom_symbol_one_hot_from_set_of_mols_featurizer(
        list_of_mols, only_mass=False, sort=True, with_other=True, as_class=False
):
    from molNet.mol.molecule import Molecule

    _possible_values = []
    for mol in list_of_mols:

        if isinstance(mol, Molecule):
            mol = mol.get_mol(with_H=True)
        if not isinstance(mol, Mol):
            continue
        for atom in mol.GetAtoms():
            s = atom.GetSymbol()
            if s not in _possible_values:
                if only_mass:
                    if atom.GetMass() <= 0:
                        continue
                _possible_values.append(s)
    if sort:
        _possible_values.sort()
    if with_other:
        _possible_values.append(None)

    if as_class:
        class Atom_CustomSymbolOneHot_Featurizer(Atom_AllSymbolOneHot_Featurizer):
            POSSIBLE_VALUES = _possible_values

        return Atom_CustomSymbolOneHot_Featurizer
    else:
        return Atom_AllSymbolOneHot_Featurizer(possible_values=_possible_values)


class Atom_HCNOPSClBrOneHot_Featurizer(Atom_AllSymbolOneHot_Featurizer):
    POSSIBLE_VALUES = ["H", "C", "N", "O", "P", "S", "Cl", "Br", None]


atom_HCNOPSClBrOneHot_featurizer = Atom_HCNOPSClBrOneHot_Featurizer()


class Atom_TotalDegreeOneHot_Featurizer(OneHotAtomFeaturizer):
    POSSIBLE_VALUES = list(range(8))

    def featurize(self, atom):
        return atom.GetTotalDegree()


atom_TotalDegreeOneHot_featurizer = Atom_TotalDegreeOneHot_Featurizer()


class Atom_DegreeOneHot_Featurizer(OneHotAtomFeaturizer):
    POSSIBLE_VALUES = list(range(8))

    def featurize(self, atom):
        return atom.GetDegree()


atom_DegreeOneHot_featurizer = Atom_DegreeOneHot_Featurizer()


class Atom_ImplicitValenceOneHot_Featurizer(OneHotAtomFeaturizer):
    POSSIBLE_VALUES = list(range(8))

    def featurize(self, atom):
        return atom.GetImplicitValence()


atom_ImplicitValenceOneHot_featurizer = Atom_ImplicitValenceOneHot_Featurizer()


class Atom_ExplicitValenceOneHot_Featurizer(OneHotAtomFeaturizer):
    POSSIBLE_VALUES = list(range(8))

    def featurize(self, atom):
        return atom.GetExplicitValence()


atom_ExplicitValenceOneHot_featurizer = Atom_ExplicitValenceOneHot_Featurizer()


class Atom_TotalValenceOneHot_Featurizer(OneHotAtomFeaturizer):
    POSSIBLE_VALUES = list(range(8))

    def featurize(self, atom):
        return atom.GetTotalValence()


atom_TotalValenceOneHot_featurizer = Atom_TotalValenceOneHot_Featurizer()


class Atom_NumRadicalElectronsOneHot_Featurizer(OneHotAtomFeaturizer):
    POSSIBLE_VALUES = list(range(5))

    def featurize(self, atom):
        return atom.GetNumRadicalElectrons()


atom_NumRadicalElectronsOneHot_featurizer = Atom_NumRadicalElectronsOneHot_Featurizer()


class Atom_FormalChargeOneHot_Featurizer(OneHotAtomFeaturizer):
    POSSIBLE_VALUES = list(range(-8, 9))

    def featurize(self, atom):
        return atom.GetFormalCharge()


atom_FormalChargeOneHot_featurizer = Atom_FormalChargeOneHot_Featurizer()


class Atom_TotalNumHsOneHot_Featurizer(OneHotAtomFeaturizer):
    POSSIBLE_VALUES = list(range(9))

    def featurize(self, atom):
        return atom.GetTotalNumHs()


atom_TotalNumHsOneHot_featurizer = Atom_TotalNumHsOneHot_Featurizer()


def atom_is_in_ring_size_n_featurizer(n, as_class=False):
    class _InRingNFeaturizer(SingleValueAtomFeaturizer):
        dtype = np.bool_

        def featurize(self, atom):
            return atom.IsInRingSize(n)

    if as_class:
        return _InRingNFeaturizer
    else:
        return _InRingNFeaturizer()


def _get_gasteiger_charge(atom):
    if not atom.HasProp("_GasteigerCharge"):
        ComputeGasteigerCharges(atom.GetOwningMol())
    gasteiger_charge = atom.GetProp("_GasteigerCharge")
    if gasteiger_charge in ["-nan", "nan", "-inf", "inf"]:
        return [0]
    return float(gasteiger_charge)


class Atom_PartialCharge_Featurizer(SingleValueAtomFeaturizer):
    dtype = np.float32
    featurize = staticmethod(_get_gasteiger_charge)


atom_PartialCharge_featurizer = Atom_PartialCharge_Featurizer()

_available_featurizer = {
    "atom_ConnectedAtoms_featurizer": atom_ConnectedAtoms_featurizer,
    "atom_HCNOPSClBrOneHot_featurizer": atom_HCNOPSClBrOneHot_featurizer,
    "atom_TotalDegreeOneHot_featurizer": atom_TotalDegreeOneHot_featurizer,
    "atom_DegreeOneHot_featurizer": atom_DegreeOneHot_featurizer,
    "atom_ImplicitValenceOneHot_featurizer": atom_ImplicitValenceOneHot_featurizer,
    "atom_ExplicitValenceOneHot_featurizer": atom_ExplicitValenceOneHot_featurizer,
    "atom_TotalValenceOneHot_featurizer": atom_TotalValenceOneHot_featurizer,
    "atom_NumRadicalElectronsOneHot_featurizer": atom_NumRadicalElectronsOneHot_featurizer,
    "atom_FormalChargeOneHot_featurizer": atom_FormalChargeOneHot_featurizer,
    "atom_TotalNumHsOneHot_featurizer": atom_TotalNumHsOneHot_featurizer,
    "atom_PartialCharge_featurizer": atom_PartialCharge_featurizer,
}

__all__ = [
    "Atom_ConnectedAtoms_Featurizer",
    "atom_ConnectedAtoms_featurizer",
    "Atom_HCNOPSClBrOneHot_Featurizer",
    "atom_HCNOPSClBrOneHot_featurizer",
    "Atom_TotalDegreeOneHot_Featurizer",
    "atom_TotalDegreeOneHot_featurizer",
    "Atom_DegreeOneHot_Featurizer",
    "atom_DegreeOneHot_featurizer",
    "Atom_ImplicitValenceOneHot_Featurizer",
    "atom_ImplicitValenceOneHot_featurizer",
    "Atom_ExplicitValenceOneHot_Featurizer",
    "atom_ExplicitValenceOneHot_featurizer",
    "Atom_TotalValenceOneHot_Featurizer",
    "atom_TotalValenceOneHot_featurizer",
    "Atom_NumRadicalElectronsOneHot_Featurizer",
    "atom_NumRadicalElectronsOneHot_featurizer",
    "Atom_FormalChargeOneHot_Featurizer",
    "atom_FormalChargeOneHot_featurizer",
    "Atom_TotalNumHsOneHot_Featurizer",
    "atom_TotalNumHsOneHot_featurizer",
    "Atom_PartialCharge_Featurizer",
    "atom_PartialCharge_featurizer",
    "atom_symbol_one_hot_from_set_of_mols_featurizer",
    "atom_is_in_ring_size_n_featurizer"

]


def get_available_featurizer():
    return _available_featurizer


def main():
    from rdkit import Chem
    from molNet.featurizer.molecule_featurizer import prepare_mol_for_featurization

    testmol = prepare_mol_for_featurization(Chem.MolFromSmiles("c1ccccc1")).GetAtoms()[
        -1
    ]
    for n, f in get_available_featurizer().items():
        print(n, end=" ")
        print(f(testmol))
    print(len(get_available_featurizer()))


if __name__ == "__main__":
    main()
