import numpy as np
from rdkit.Chem.rdMolDescriptors import (
    GetUSR,
    GetUSRCAT,
    CalcAUTOCORR2D,
    GetConnectivityInvariants,
    BCUT2D,
    GetFeatureInvariants,
    CalcEEMcharges,
    CalcMORSE,
    CalcCrippenDescriptors,
    CalcRDF,
    CalcWHIM,
    CalcAUTOCORR3D,
)

from molNet.featurizer._molecule_featurizer import (
    FixedSizeMoleculeFeaturizer,
    VarSizeMoleculeFeaturizer,
)


class AUTOCORR2D_Featurizer(FixedSizeMoleculeFeaturizer):
    # statics
    LENGTH = 192
    dtype = np.float32
    featurize = staticmethod(CalcAUTOCORR2D)
    # normalization
    # functions


class AUTOCORR3D_Featurizer(FixedSizeMoleculeFeaturizer):
    # statics
    LENGTH = 80
    dtype = np.float32
    featurize = staticmethod(CalcAUTOCORR3D)
    # normalization
    # functions


class BCUT2D_Featurizer(FixedSizeMoleculeFeaturizer):
    # statics
    LENGTH = 8
    dtype = np.float32
    featurize = staticmethod(BCUT2D)
    # normalization
    # functions


class ConnectivityInvariants_Featurizer(VarSizeMoleculeFeaturizer):
    # statics
    dtype = np.int64

    featurize = staticmethod(GetConnectivityInvariants)
    # normalization
    # functions


class CrippenDescriptors_Featurizer(FixedSizeMoleculeFeaturizer):
    # statics
    LENGTH = 2
    dtype = np.float32
    featurize = staticmethod(CalcCrippenDescriptors)
    # normalization
    # functions


class EEMcharges_Featurizer(VarSizeMoleculeFeaturizer):
    # statics
    dtype = np.float64

    featurize = staticmethod(CalcEEMcharges)
    # normalization
    # functions


class FeatureInvariants_Featurizer(VarSizeMoleculeFeaturizer):
    # statics
    dtype = np.int32

    featurize = staticmethod(GetFeatureInvariants)
    # normalization
    # functions


from rdkit.Chem.rdMolDescriptors import CalcGETAWAY as _oCalcGETAWAY
from rdkit.Chem import GetMolFrags


def CalcGETAWAY(mol):
    frags = GetMolFrags(mol, asMols=True)
    if len(frags) > 1:
        frags = sorted(frags, key=lambda m: mol.GetNumAtoms(), reverse=True)
        return _oCalcGETAWAY(frags[0])
    return _oCalcGETAWAY(mol)


class GETAWAY_Featurizer(FixedSizeMoleculeFeaturizer):
    # statics
    LENGTH = 273
    dtype = np.float32
    featurize = staticmethod(CalcGETAWAY)
    # normalization
    # functions


class MORSE_Featurizer(FixedSizeMoleculeFeaturizer):
    # statics
    LENGTH = 224
    dtype = np.float32
    featurize = staticmethod(CalcMORSE)
    # normalization
    # functions


class RDF_Featurizer(FixedSizeMoleculeFeaturizer):
    # statics
    LENGTH = 210
    dtype = np.float32
    featurize = staticmethod(CalcRDF)
    # normalization
    # functions


class USR_Featurizer(FixedSizeMoleculeFeaturizer):
    # statics
    LENGTH = 12
    dtype = np.float32
    featurize = staticmethod(GetUSR)
    # normalization
    # functions


class USRCAT_Featurizer(FixedSizeMoleculeFeaturizer):
    # statics
    LENGTH = 60
    dtype = np.float32
    featurize = staticmethod(GetUSRCAT)
    # normalization
    # functions


class WHIM_Featurizer(FixedSizeMoleculeFeaturizer):
    # statics
    LENGTH = 114
    dtype = np.float32
    featurize = staticmethod(CalcWHIM)
    # normalization
    # functions


molecule_AUTOCORR2D = AUTOCORR2D_Featurizer()
molecule_AUTOCORR3D = AUTOCORR3D_Featurizer()
molecule_BCUT2D = BCUT2D_Featurizer()
molecule_ConnectivityInvariants = ConnectivityInvariants_Featurizer()
molecule_CrippenDescriptors = CrippenDescriptors_Featurizer()
molecule_EEMcharges = EEMcharges_Featurizer()
molecule_FeatureInvariants = FeatureInvariants_Featurizer()
molecule_GETAWAY = GETAWAY_Featurizer()
molecule_MORSE = MORSE_Featurizer()
molecule_RDF = RDF_Featurizer()
molecule_USR = USR_Featurizer()
molecule_USRCAT = USRCAT_Featurizer()
molecule_WHIM = WHIM_Featurizer()

_available_featurizer = {
    "molecule_AUTOCORR2D": molecule_AUTOCORR2D,
    "molecule_AUTOCORR3D": molecule_AUTOCORR3D,
    "molecule_BCUT2D": molecule_BCUT2D,
    "molecule_ConnectivityInvariants": molecule_ConnectivityInvariants,
    "molecule_CrippenDescriptors": molecule_CrippenDescriptors,
    "molecule_EEMcharges": molecule_EEMcharges,
    "molecule_FeatureInvariants": molecule_FeatureInvariants,
    "molecule_GETAWAY": molecule_GETAWAY,
    "molecule_MORSE": molecule_MORSE,
    "molecule_RDF": molecule_RDF,
    "molecule_USR": molecule_USR,
    "molecule_USRCAT": molecule_USRCAT,
    "molecule_WHIM": molecule_WHIM,
}


def get_available_featurizer():
    return _available_featurizer


__all__ = [
    "AUTOCORR2D_Featurizer",
    "AUTOCORR3D_Featurizer",
    "BCUT2D_Featurizer",
    "ConnectivityInvariants_Featurizer",
    "CrippenDescriptors_Featurizer",
    "EEMcharges_Featurizer",
    "FeatureInvariants_Featurizer",
    "GETAWAY_Featurizer",
    "MORSE_Featurizer",
    "RDF_Featurizer",
    "USR_Featurizer",
    "USRCAT_Featurizer",
    "WHIM_Featurizer",
    "molecule_AUTOCORR2D",
    "molecule_AUTOCORR3D",
    "molecule_BCUT2D",
    "molecule_ConnectivityInvariants",
    "molecule_CrippenDescriptors",
    "molecule_EEMcharges",
    "molecule_FeatureInvariants",
    "molecule_GETAWAY",
    "molecule_MORSE",
    "molecule_RDF",
    "molecule_USR",
    "molecule_USRCAT",
    "molecule_WHIM",
]


def main():
    from rdkit import Chem

    testmol = Chem.MolFromSmiles("c1ccccc1")
    for n, f in get_available_featurizer().items():
        print(n, f(testmol))


if __name__ == "__main__":
    main()
