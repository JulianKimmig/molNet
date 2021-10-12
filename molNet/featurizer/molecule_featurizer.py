from rdkit.Chem.rdmolops import GetMolFrags

from ._autogen_molecule_featurizer import *
from ._autogen_molecule_featurizer import (
    MoleculeFeaturizer,
    SingleValueMoleculeFeaturizer,
)
from .featurizer import FeaturizerList


class ExtendMolnetFeaturizer(MoleculeFeaturizer):
    def featurize(self, mol):
        r = mol.molnet_features if hasattr(mol, "molnet_features") else []
        return np.array([r]).flatten()


extend_molnet_featurizer = ExtendMolnetFeaturizer(name="extend_molnet_featurizer")


class NumAtoms_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.int32
    # normalization
    linear_norm_parameter = (
        0.0028101191309833906,
        0.4586559034152283,
    )  # error of 2.08E-01 with sample range (2.00E+00,9.38E+02) resulting in fit range (4.64E-01,3.09E+00)
    min_max_norm_parameter = (
        22.699274586285846,
        70.39472667366697,
    )  # error of 2.45E-02 with sample range (2.00E+00,9.38E+02) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (
        46.31674761573806,
        0.10711279271638556,
    )  # error of 1.75E-02 with sample range (2.00E+00,9.38E+02) resulting in fit range (8.60E-03,1.00E+00)
    dual_sigmoidal_norm_parameter = (
        43.70797609285935,
        0.1511085487560505,
        0.07900109330436363,
    )  # error of 9.28E-03 with sample range (2.00E+00,9.38E+02) resulting in fit range (1.83E-03,1.00E+00)
    genlog_norm_parameter = (
        0.07326085064819542,
        -44.92533678413176,
        0.022954022788335537,
        4.515441426070382e-05,
    )  # error of 8.12E-03 with sample range (2.00E+00,9.38E+02) resulting in fit range (8.09E-08,1.00E+00)
    preferred_normalization = "genlog"

    # functions
    def featurize(self, mol):
        return mol.GetNumAtoms()


molecule_num_atoms = NumAtoms_Featurizer()


class NumBonds_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.int32
    # normalization
    linear_norm_parameter = (
        0.002743260112020973,
        0.45085043244012823,
    )  # error of 2.10E-01 with sample range (0.00E+00,9.97E+02) resulting in fit range (4.51E-01,3.19E+00)
    min_max_norm_parameter = (
        22.92431994469887,
        73.53427413100934,
    )  # error of 2.46E-02 with sample range (0.00E+00,9.97E+02) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (
        48.01236350848081,
        0.10071768927745786,
    )  # error of 1.76E-02 with sample range (0.00E+00,9.97E+02) resulting in fit range (7.88E-03,1.00E+00)
    dual_sigmoidal_norm_parameter = (
        45.26546665522718,
        0.1416051615875594,
        0.07451723131457108,
    )  # error of 9.14E-03 with sample range (0.00E+00,9.97E+02) resulting in fit range (1.64E-03,1.00E+00)
    genlog_norm_parameter = (
        0.06893983960267623,
        -54.16701644496038,
        0.03480866141021429,
        4.781043915143292e-05,
    )  # error of 7.91E-03 with sample range (0.00E+00,9.97E+02) resulting in fit range (2.81E-08,1.00E+00)
    preferred_normalization = "genlog"

    # functions
    def featurize(self, mol):
        return mol.GetNumBonds()


molecule_num_bonds = NumBonds_Featurizer()


class NumFragments_Featurizer(SingleValueMoleculeFeaturizer):
    # statics
    dtype = np.int32
    # normalization
    linear_norm_parameter = (
        0.007725304720410886,
        0.95870371666541,
    )  # error of 7.29E-03 with sample range (1.00E+00,5.80E+01) resulting in fit range (9.66E-01,1.41E+00)
    min_max_norm_parameter = (
        1.0000000095197468,
        2.0373839527585793,
    )  # error of 2.36E-03 with sample range (1.00E+00,5.80E+01) resulting in fit range (0.00E+00,1.00E+00)
    sigmoidal_norm_parameter = (
        -0.5414310124721092,
        1.294708062950411,
    )  # error of 8.38E-04 with sample range (1.00E+00,5.80E+01) resulting in fit range (8.80E-01,1.00E+00)
    dual_sigmoidal_norm_parameter = (
        -0.5414310124721092,
        1.0,
        1.294708062950411,
    )  # error of 8.38E-04 with sample range (1.00E+00,5.80E+01) resulting in fit range (8.80E-01,1.00E+00)
    genlog_norm_parameter = (
        1.2824325546342243,
        -3.7454818331663993,
        0.39329927649044794,
        0.006786382098985755,
    )  # error of 8.26E-04 with sample range (1.00E+00,5.80E+01) resulting in fit range (8.77E-01,1.00E+00)
    preferred_normalization = "min_max"

    # functions

    def featurize(self, mol):
        return len(GetMolFrags(mol))


molecule_num_fragments = NumFragments_Featurizer()

from ._autogen_molecule_featurizer import _available_featurizer as _agaf

_available_featurizer = [
    *_agaf,
    molecule_num_atoms,
    molecule_num_bonds,
    molecule_num_fragments,
]


class AllSingleValueMoleculeFeaturizer(FeaturizerList):
    dtype = np.float32

    def __init__(self, *args, **kwargs):
        super().__init__(
            [
                f
                for f in _available_featurizer
                if isinstance(f, SingleValueMoleculeFeaturizer)
            ],
            *args,
            **kwargs
        )


molecule_all_single_val_feats = AllSingleValueMoleculeFeaturizer()

default_molecule_featurizer = FeaturizerList(
    [],
    name="default_molecule_featurizer",
)
