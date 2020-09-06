import rdkit
import rdkit.Chem.AllChem
from rdkit.Chem import rdchem

from .featurizer import OneHotFeaturizer, FeaturizerList, LambdaFeaturizer

__all__ = ['atom_atomic_number',
           'atom_atomic_number_one_hot',
           'atom_chiral_tag_one_hot',
           'atom_degree',
           'atom_degree_one_hot',
           'atom_explicit_valence',
           'atom_explicit_valence_one_hot',
           'atom_formal_charge',
           'atom_formal_charge_one_hot',
           'atom_hybridization_one_hot',
           'atom_implicit_valence',
           'atom_implicit_valence_one_hot',
           'atom_is_aromatic',
           'atom_is_in_ring',
           'atom_mass',
           'atom_num_radical_electrons',
           'atom_num_radical_electrons_one_hot',
           'atom_partial_charge',
           'atom_symbol_one_hot',
           'atom_total_degree',
           'atom_total_degree_one_hot',
           'atom_total_num_H',
           'atom_total_num_H_one_hot',
           'default_atom_featurizer',
           ]

atom_symbol_one_hot = OneHotFeaturizer(
    possible_values=['O', 'Si', 'Al', 'Fe', 'Ca', 'Na', 'Mg', 'K', 'Ti', 'H', 'P', 'Mn', 'F', 'Sr', 'S', 'C', 'Zr',
                     'Cl', 'V', 'Cr', 'Rb', 'Ni', 'Zn', 'Cu', 'Y', 'Co', 'Sc', 'Li', 'Nb', 'N', 'Ga', 'B', 'Ar', 'Be',
                     'Br', 'As', 'Ge', 'Mo', 'Kr', 'Se', 'He', 'Ne', 'Tc', 'Ba', 'Ce', 'Nd', 'La', 'Pb', 'Pr', 'Sm',
                     'Re', 'Gd', 'Dy', 'Rn', 'Er', 'Yb', 'Xe', 'Cs', 'Hf', 'At', 'Sn', 'Pm', 'Eu', 'Ta', 'Po', 'Ho',
                     'W', 'Tb', 'Tl', 'Lu', 'Tm', 'I', 'In', 'Sb', 'Cd', 'Hg', 'Ag', 'Pd', 'Bi', 'Pt', 'Au', 'Os', 'Ru',
                     'Rh', 'Te', 'Ir', 'Fr', 'Th', 'Ra', 'Ac', 'U', 'Pa', 'Np', 'Pu', None],
    pre_featurize=lambda atom: atom.GetSymbol(),
    name="atom_symbol_one_hot"
)

atom_symbol_hcnopsclbr_one_hot = OneHotFeaturizer(
    possible_values=['H', 'C', 'N', 'O', 'P', 'S', 'Cl', 'Br'],
    pre_featurize=lambda atom: atom.GetSymbol(),
    name="atom_symbol_one_hot"
)

atom_atomic_number_one_hot = OneHotFeaturizer(
    possible_values=list(range(1, 119)),
    pre_featurize=lambda atom: atom.GetAtomicNum(),
    name="atomic_number_one_hot"
)
atom_atomic_number = LambdaFeaturizer(lambda atom: [atom.GetAtomicNum()], length=1,
                                      name="atomic_number")

atom_total_degree_one_hot = OneHotFeaturizer(
    possible_values=list(range(8)),
    pre_featurize=lambda atom: atom.GetTotalDegree(),
    name="atom_total_degree_one_hot"
)
atom_total_degree = LambdaFeaturizer(lambda atom: [atom.GetTotalDegree()], length=1,
                                     name="atom_total_degree")

atom_degree_one_hot = OneHotFeaturizer(
    possible_values=list(range(8)),
    pre_featurize=lambda atom: atom.GetDegree(),
    name="atom_degree_one_hot"
)
atom_degree = LambdaFeaturizer(lambda atom: [atom.GetDegree()], length=1,
                               name="atom_degree")

atom_implicit_valence_one_hot = OneHotFeaturizer(
    possible_values=list(range(8)),
    pre_featurize=lambda atom: atom.GetImplicitValence(),
    name="atom_implicit_valence_one_hot"
)

atom_implicit_valence = LambdaFeaturizer(lambda atom: [atom.GetImplicitValence()], length=1,
                                         name="atom_implicit_valence")

atom_explicit_valence_one_hot = OneHotFeaturizer(
    possible_values=list(range(8)),
    pre_featurize=lambda atom: atom.GetExplicitValence(),
    name="atom_explicit_valence_one_hot"
)

atom_explicit_valence = LambdaFeaturizer(lambda atom: [atom.GetExplicitValence()], length=1,
                                         name="atom_explicit_valence")

atom_num_radical_electrons = LambdaFeaturizer(lambda atom: [atom.GetNumRadicalElectrons()], length=1,
                                              name="atom_num_radical_electrons")

atom_num_radical_electrons_one_hot = OneHotFeaturizer(
    possible_values=list(range(5)),
    pre_featurize=lambda atom: atom.GetNumRadicalElectrons(),
    name="atom_num_radical_electrons_one_hot"
)

atom_hybridization_one_hot = OneHotFeaturizer(
    possible_values=[rdkit.Chem.rdchem.HybridizationType.SP,
                     rdkit.Chem.rdchem.HybridizationType.SP2,
                     rdkit.Chem.rdchem.HybridizationType.SP3,
                     rdkit.Chem.rdchem.HybridizationType.SP3D,
                     rdkit.Chem.rdchem.HybridizationType.SP3D2,
                     rdkit.Chem.rdchem.HybridizationType.S,
                     rdkit.Chem.rdchem.HybridizationType.OTHER,
                     rdkit.Chem.rdchem.HybridizationType.UNSPECIFIED,
                     ],
    pre_featurize=lambda atom: atom.GetHybridization(),
    name="atom_hybridization_one_hot"
)

atom_chiral_tag_one_hot = OneHotFeaturizer(
    possible_values=[rdkit.Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
                     rdkit.Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
                     rdkit.Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
                     rdkit.Chem.rdchem.ChiralType.CHI_OTHER
                     ],
    pre_featurize=lambda atom: atom.GetChiralTag(),
    name="atom_chiral_tag_one_hot"
)

atom_formal_charge_one_hot = OneHotFeaturizer(
    possible_values=list(range(-8, 9)),
    pre_featurize=lambda atom: atom.GetFormalCharge(),
    name="atom_formal_charge_one_hot"
)

atom_formal_charge = LambdaFeaturizer(lambda atom: [atom.GetFormalCharge()], length=1,
                                      name="atom_formal_charge")

atom_mass = LambdaFeaturizer(lambda atom: [atom.GetMass() * 0.01], length=1,
                             name="atom_mass")

atom_total_num_H_one_hot = OneHotFeaturizer(
    possible_values=list(range(9)),
    pre_featurize=lambda atom: atom.GetTotalNumHs(),
    name="atom_total_num_H_one_hot"
)

atom_total_num_H = LambdaFeaturizer(lambda atom: [atom.GetTotalNumHs()], length=1,
                                    name="atom_total_num_H")

atom_is_aromatic = LambdaFeaturizer(
    lambda atom: [atom.GetIsAromatic()], length=1,
    name="atom_is_aromatic",
)

atom_is_in_ring = LambdaFeaturizer(lambda atom: [atom.IsInRing()], length=1,
                                   name="atom_is_in_ring")


def _get_gasteiger_charge(atom):
    if not atom.HasProp('_GasteigerCharge'):
        rdkit.Chem.AllChem.ComputeGasteigerCharges(atom.GetOwningMol())
    gasteiger_charge = atom.GetProp('_GasteigerCharge')
    if gasteiger_charge in ['-nan', 'nan', '-inf', 'inf']:
        return [0]
    return [float(gasteiger_charge)]


atom_partial_charge = LambdaFeaturizer(_get_gasteiger_charge, length=1,
                                       name="atom_partial_charge")

default_atom_featurizer = FeaturizerList([
    atom_symbol_one_hot,
    atom_formal_charge,
    atom_partial_charge,
    atom_mass,
    atom_total_degree_one_hot,
    atom_degree_one_hot,
    atom_hybridization_one_hot,
],
    name="default_atom_featurizer")
