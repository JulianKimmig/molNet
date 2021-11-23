import numpy as np
import rdkit
import rdkit.Chem.AllChem
from rdkit.Chem import SetHybridization, HybridizationType
from rdkit.Chem.rdchem import Mol

from molNet import MOLNET_LOGGER
from molNet.featurizer._atom_featurizer import SingleValueAtomFeaturizer
from molNet.featurizer.featurizer import FeaturizerList

_available_featurizer = {}
__all__ = []
try:
    from molNet.featurizer import _manual_atom_featurizer
    from molNet.featurizer._manual_molecule_featurizer import *

    for n, f in _manual_atom_featurizer.get_available_featurizer().items():
        if n in _available_featurizer:
            MOLNET_LOGGER.warning(f"encoutered duplicate while collecting moelcule featurizer: {n}")
            continue
        _available_featurizer[n] = f

    __all__ += _manual_atom_featurizer.__all__
except Exception as e:
    MOLNET_LOGGER.exception(e)

try:
    from molNet.featurizer import _autogen_atom_featurizer
    from molNet.featurizer._autogen_atom_featurizer import *

    for n, f in _autogen_atom_featurizer.get_available_featurizer().items():
        if n in _available_featurizer:
            n = f"autogen_molecule_featurizer_{n}"
        if n in _available_featurizer:
            MOLNET_LOGGER.warning(f"encoutered duplicate while collecting moelcule featurizer: {n}")
            continue
        _available_featurizer[n] = f

    __all__ += _autogen_atom_featurizer.__all__
except Exception as e:
    MOLNET_LOGGER.exception(e)


class AllSingleValueAtomFeaturizer(FeaturizerList):
    dtype = np.float32

    def __init__(self, *args, **kwargs):
        super().__init__(
            [
                f
                for n, f in _available_featurizer.items()
                if isinstance(f, SingleValueAtomFeaturizer)
            ],
            *args,
            **kwargs
        )


atom_all_single_val_feats = AllSingleValueAtomFeaturizer(name="atom_all_single_val_feats")
__all__.extend(["atom_all_single_val_feats", "AllSingleValueAtomFeaturizer"])
_available_featurizer["atom_all_single_val_feats"] = atom_all_single_val_feats


def get_available_featurizer():
    return _available_featurizer


def main():
    from rdkit import Chem

    testmol = Chem.MolFromSmiles("c1ccccc1").GetAtoms()[0]

    for n, f in get_available_featurizer().items():
        print(n, end=" ")
        print(f(testmol))
    print(len(get_available_featurizer()))


if __name__ == "__main__":
    main()

def atom_symbol_one_hot_from_set(
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
            possible_values = _possible_values

        return Atom_CustomSymbolOneHot_Featurizer
    else:
        return Atom_AllSymbolOneHot_Featurizer(possible_values=_possible_values)


atom_symbol_hcnopsclbr_one_hot = Atom_AllSymbolOneHot_Featurizer(
    possible_values=["H", "C", "N", "O", "P", "S", "Cl", "Br", None])(

    atom_atomic_number_one_hot=OneHotFeaturizer(
        possible_values=list(range(1, 119)),
        pre_featurize=_get_atom_num,
        name="atomic_number_one_hot",
    )

atom_total_degree_one_hot = OneHotFeaturizer(
    possible_values=list(range(8)),
    pre_featurize=_get_atom_tot_deg,
    name="atom_total_degree_one_hot",
)


atom_degree_one_hot = OneHotFeaturizer(
    possible_values=list(range(8)),
    pre_featurize=_get_atom_deg,
    name="atom_degree_one_hot",
)



atom_implicit_valence_one_hot = OneHotFeaturizer(
    possible_values=list(range(8)),
    pre_featurize=_get_atom_imp_val,
    name="atom_implicit_valence_one_hot",
)

atom_explicit_valence_one_hot = OneHotFeaturizer(
    possible_values=list(range(8)),
    pre_featurize=_get_atom_exp_val,
    name="atom_explicit_valence_one_hot",
)



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


atom_formal_charge_one_hot = OneHotFeaturizer(
    possible_values=list(range(-8, 9)),
    pre_featurize=_get_atom_formal_charge,
    name="atom_formal_charge_one_hot",
)

atom_total_num_H_one_hot = OneHotFeaturizer(
    possible_values=list(range(9)),
    pre_featurize=_get_atom_tot_h,
    name="atom_total_num_H_one_hot",
)




def atom_is_in_ring_size_n(n):
    class _InRingNFeaturizer(SingleValueAtomFeaturizer):
        dtype = np.bool_

        def featurize(self, atom):
            return [atom.IsInRingSize(n)]

    return _InRingNFeaturizer()


def atom_is_in_ring_size_n_to_m_one_hot(n: int, m: int):
    if m <= n:
        raise ValueError("m has to be larger than n")
    return FeaturizerList(
        [atom_is_in_ring_size_n(i) for i in range(n, m + 1)],
        name="atom_is_in_ring_size_{}_to_{}_one_hot".format(n, m),
    )


def _get_gasteiger_charge(atom):
    if not atom.HasProp("_GasteigerCharge"):
        rdkit.Chem.AllChem.ComputeGasteigerCharges(atom.GetOwningMol())
    gasteiger_charge = atom.GetProp("_GasteigerCharge")
    if gasteiger_charge in ["-nan", "nan", "-inf", "inf"]:
        return [0]
    return [float(gasteiger_charge)]


class AtomPartialChargeFeaturizer(SingleValueAtomFeaturizer):
    dtype = np.float32
    featurize = staticmethod(_get_gasteiger_charge)


atom_partial_charge = AtomPartialChargeFeaturizer()
