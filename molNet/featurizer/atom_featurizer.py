import numpy as np
import rdkit
import rdkit.Chem.AllChem
from rdkit.Chem import rdchem, SetHybridization, HybridizationType
from rdkit.Chem.rdchem import Mol

from ._atom_featurizer import SingleValueAtomFeaturizer
from .featurizer import OneHotFeaturizer, FeaturizerList

__all__ = [
    "atom_atomic_number",
    "atom_atomic_number_one_hot",
    "atom_chiral_tag_one_hot",
    "atom_degree",
    "atom_degree_one_hot",
    "atom_explicit_valence",
    "atom_explicit_valence_one_hot",
    "atom_formal_charge",
    "atom_formal_charge_one_hot",
    "atom_hybridization_one_hot",
    "atom_implicit_valence",
    "atom_implicit_valence_one_hot",
    "atom_is_aromatic",
    "atom_is_in_ring",
    "atom_mass",
    "atom_num_radical_electrons",
    "atom_num_radical_electrons_one_hot",
    "atom_partial_charge",
    "atom_symbol_one_hot",
    "atom_total_degree",
    "atom_total_degree_one_hot",
    "atom_total_num_H",
    "atom_total_num_H_one_hot",
    "default_atom_featurizer",
]


def _get_atom_symbol(atom):
    return atom.GetSymbol()


def _get_atom_num(atom):
    return atom.GetAtomicNum()


def _get_atom_tot_deg(atom):
    return atom.GetTotalDegree()


def _get_atom_deg(atom):
    return atom.GetDegree()


def _get_atom_imp_val(atom):
    return atom.GetImplicitValence()


def _get_atom_exp_val(atom):
    return atom.GetExplicitValence()


def _get_atom_num_rad_el(atom):
    return atom.GetNumRadicalElectrons()


def _get_atom_chiral_tag(atom):
    return atom.GetChiralTag()


def _get_atom_in_ring(atom):
    return atom.IsInRing()


def _get_atom_formal_charge(atom):
    return atom.GetFormalCharge()


def _get_atom_get_mass(atom):
    return atom.GetMass()


def _get_atom_tot_h(atom):
    return atom.GetTotalNumHs()


def _get_atom_is_arom(atom):
    return atom.GetIsAromatic()


# class _to_array:
#    def __init__(self, funcs):
#        self.funcs = funcs

#    def __call__(self, atom):
#        return [f(atom) for f in self.funcs]


# def as_array(*funcs):
#    return _to_array(funcs)


atom_symbol_one_hot = OneHotFeaturizer(
    possible_values=[
        "O",
        "Si",
        "Al",
        "Fe",
        "Ca",
        "Na",
        "Mg",
        "K",
        "Ti",
        "H",
        "P",
        "Mn",
        "F",
        "Sr",
        "S",
        "C",
        "Zr",
        "Cl",
        "V",
        "Cr",
        "Rb",
        "Ni",
        "Zn",
        "Cu",
        "Y",
        "Co",
        "Sc",
        "Li",
        "Nb",
        "N",
        "Ga",
        "B",
        "Ar",
        "Be",
        "Br",
        "As",
        "Ge",
        "Mo",
        "Kr",
        "Se",
        "He",
        "Ne",
        "Tc",
        "Ba",
        "Ce",
        "Nd",
        "La",
        "Pb",
        "Pr",
        "Sm",
        "Re",
        "Gd",
        "Dy",
        "Rn",
        "Er",
        "Yb",
        "Xe",
        "Cs",
        "Hf",
        "At",
        "Sn",
        "Pm",
        "Eu",
        "Ta",
        "Po",
        "Ho",
        "W",
        "Tb",
        "Tl",
        "Lu",
        "Tm",
        "I",
        "In",
        "Sb",
        "Cd",
        "Hg",
        "Ag",
        "Pd",
        "Bi",
        "Pt",
        "Au",
        "Os",
        "Ru",
        "Rh",
        "Te",
        "Ir",
        "Fr",
        "Th",
        "Ra",
        "Ac",
        "U",
        "Pa",
        "Np",
        "Pu",
        None,
    ],
    pre_featurize=_get_atom_symbol,
    name="atom_symbol_one_hot",
)


def atom_symbol_one_hot_from_set(
    list_of_mols, only_mass=False, sort=True, with_other=True
):
    from molNet.mol.molecule import Molecule

    possible_values = []
    for mol in list_of_mols:

        if isinstance(mol, Molecule):
            mol = mol.get_mol(with_H=True)
        if not isinstance(mol, Mol):
            continue
        for atom in mol.GetAtoms():
            s = atom.GetSymbol()
            if s not in possible_values:
                if only_mass:
                    if atom.GetMass() <= 0:
                        continue
                possible_values.append(s)
    if sort:
        possible_values.sort()
    if with_other:
        possible_values.append(None)
    return OneHotFeaturizer(
        possible_values=possible_values,
        pre_featurize=_get_atom_symbol,
        name="custom_atom_symbol_one_hot",
    )


atom_symbol_hcnopsclbr_one_hot = OneHotFeaturizer(
    possible_values=["H", "C", "N", "O", "P", "S", "Cl", "Br", None],
    pre_featurize=_get_atom_symbol,
    name="atom_symbol_one_hot",
)


atom_atomic_number_one_hot = OneHotFeaturizer(
    possible_values=list(range(1, 119)),
    pre_featurize=_get_atom_num,
    name="atomic_number_one_hot",
)


class AtomicNumberFeaturizer(SingleValueAtomFeaturizer):
    dtype = np.int32
    featurize_function = staticmethod(_get_atom_num)


atom_atomic_number = AtomicNumberFeaturizer()


atom_total_degree_one_hot = OneHotFeaturizer(
    possible_values=list(range(8)),
    pre_featurize=_get_atom_tot_deg,
    name="atom_total_degree_one_hot",
)


class AtomTotalDegreeFeaturizer(SingleValueAtomFeaturizer):
    dtype = np.int32
    featurize_function = staticmethod(_get_atom_tot_deg)


atom_total_degree = AtomTotalDegreeFeaturizer()

atom_degree_one_hot = OneHotFeaturizer(
    possible_values=list(range(8)),
    pre_featurize=_get_atom_deg,
    name="atom_degree_one_hot",
)


class AtomDegreeFeaturizer(SingleValueAtomFeaturizer):
    dtype = np.int32
    featurize_function = staticmethod(_get_atom_deg)


atom_degree = AtomTotalDegreeFeaturizer()

atom_implicit_valence_one_hot = OneHotFeaturizer(
    possible_values=list(range(8)),
    pre_featurize=_get_atom_imp_val,
    name="atom_implicit_valence_one_hot",
)


class AtomImplicitValenceFeaturizer(SingleValueAtomFeaturizer):
    dtype = np.int32
    featurize_function = staticmethod(_get_atom_imp_val)


atom_implicit_valence = AtomImplicitValenceFeaturizer()

atom_explicit_valence_one_hot = OneHotFeaturizer(
    possible_values=list(range(8)),
    pre_featurize=_get_atom_exp_val,
    name="atom_explicit_valence_one_hot",
)


class AtomExplicitValenceFeaturizer(SingleValueAtomFeaturizer):
    dtype = np.int32
    featurize_function = staticmethod(_get_atom_exp_val)


atom_explicit_valence = AtomExplicitValenceFeaturizer()


class AtomNumRadicalElectronsFeaturizer(SingleValueAtomFeaturizer):
    dtype = np.int32
    featurize_function = staticmethod(_get_atom_num_rad_el)


atom_num_radical_electrons = AtomNumRadicalElectronsFeaturizer()


atom_num_radical_electrons_one_hot = OneHotFeaturizer(
    possible_values=list(range(5)),
    pre_featurize=_get_atom_num_rad_el,
    name="atom_num_radical_electrons_one_hot",
)


def get_assured_hybridization(atom):
    h = atom.GetHybridization()
    if h == HybridizationType.UNSPECIFIED:
        SetHybridization(atom.GetOwningMol())
        h = atom.GetHybridization()
    return h


atom_hybridization_one_hot = OneHotFeaturizer(
    possible_values=[
        HybridizationType.SP,
        HybridizationType.SP2,
        HybridizationType.SP3,
        HybridizationType.SP3D,
        HybridizationType.SP3D2,
        HybridizationType.S,
        HybridizationType.OTHER,
        HybridizationType.UNSPECIFIED,
    ],
    feature_descriptions=[
        "hybridization SP",
        "hybridization SP2",
        "hybridization SP3",
        "hybridization SP3D",
        "hybridization SP3D2",
        "hybridization S",
        "hybridization OTHER",
        "hybridization UNSPECIFIED",
    ],
    pre_featurize=get_assured_hybridization,
    name="atom_hybridization_one_hot",
)

atom_chiral_tag_one_hot = OneHotFeaturizer(
    possible_values=[
        rdkit.Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        rdkit.Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        rdkit.Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        rdkit.Chem.rdchem.ChiralType.CHI_OTHER,
    ],
    pre_featurize=_get_atom_chiral_tag,
    name="atom_chiral_tag_one_hot",
)

atom_formal_charge_one_hot = OneHotFeaturizer(
    possible_values=list(range(-8, 9)),
    pre_featurize=_get_atom_formal_charge,
    name="atom_formal_charge_one_hot",
)


class AtomFromalChargeFeaturizer(SingleValueAtomFeaturizer):
    dtype = np.float32
    featurize_function = staticmethod(_get_atom_formal_charge)


atom_formal_charge = AtomFromalChargeFeaturizer()


class AtomMassFeaturizer(SingleValueAtomFeaturizer):
    dtype = np.float32
    featurize_function = staticmethod(_get_atom_get_mass)


atom_mass = AtomMassFeaturizer()

atom_total_num_H_one_hot = OneHotFeaturizer(
    possible_values=list(range(9)),
    pre_featurize=_get_atom_tot_h,
    name="atom_total_num_H_one_hot",
)


class AtomTotalNumHsFeaturizer(SingleValueAtomFeaturizer):
    dtype = np.int32
    featurize_function = staticmethod(_get_atom_tot_h)


atom_total_num_H = AtomTotalNumHsFeaturizer()


class AtomIsAromaticFeaturizer(SingleValueAtomFeaturizer):
    dtype = np.bool_
    featurize_function = staticmethod(_get_atom_is_arom)


atom_is_aromatic = AtomIsAromaticFeaturizer()


def atom_is_in_ring_size_n(n):
    class _InRingNFeaturizer(SingleValueAtomFeaturizer):
        dtype = np.bool_

        def featurize_function(self, atom):
            return [atom.IsInRingSize(n)]

    return _InRingNFeaturizer()


def atom_is_in_ring_size_n_to_m_one_hot(n: int, m: int):
    if m <= n:
        raise ValueError("m has to be larger than n")
    return FeaturizerList(
        [atom_is_in_ring_size_n(i) for i in range(n, m + 1)],
        name="atom_is_in_ring_size_{}_to_{}_one_hot".format(n, m),
    )


class AtomIsInRingFeaturizer(SingleValueAtomFeaturizer):
    dtype = np.bool_
    featurize_function = staticmethod(_get_atom_in_ring)


atom_is_in_ring = AtomIsInRingFeaturizer()


def _get_gasteiger_charge(atom):
    if not atom.HasProp("_GasteigerCharge"):
        rdkit.Chem.AllChem.ComputeGasteigerCharges(atom.GetOwningMol())
    gasteiger_charge = atom.GetProp("_GasteigerCharge")
    if gasteiger_charge in ["-nan", "nan", "-inf", "inf"]:
        return [0]
    return [float(gasteiger_charge)]


class AtomPartialChargeFeaturizer(SingleValueAtomFeaturizer):
    dtype = np.float32
    featurize_function = staticmethod(_get_gasteiger_charge)


atom_partial_charge = AtomPartialChargeFeaturizer()


default_atom_featurizer = FeaturizerList(
    [
        atom_symbol_one_hot,
        atom_formal_charge,
        atom_partial_charge,
        atom_mass,
        atom_total_degree_one_hot,
        atom_degree_one_hot,
        atom_hybridization_one_hot,
        atom_num_radical_electrons,
        atom_is_aromatic,
    ],
    name="default_atom_featurizer",
)
