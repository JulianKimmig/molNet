from molNet.featurizer._molecule_featurizer import (
    MoleculeFeaturizer,
    SingleValueMoleculeFeaturizer,
)
from molNet.featurizer.featurizer import FixedSizeFeaturizer
import numpy as np
from numpy import inf, nan
from rdkit.DataStructs.cDataStructs import ConvertToNumpyArray
from rdkit.Chem.rdMolDescriptors import (
    CalcCrippenDescriptors,
    GetFeatureInvariants,
    CalcEEMcharges,
    CalcAUTOCORR2D,
    GetUSRCAT,
    CalcWHIM,
    CalcGETAWAY,
    CalcMORSE,
    BCUT2D,
    GetUSR,
    CalcAUTOCORR3D,
    CalcRDF,
    GetConnectivityInvariants,
)


class AUTOCORR2D_Featurizer(FixedSizeFeaturizer, MoleculeFeaturizer):
    # statics
    LENGTH = 192
    dtype = np.float32
    featurize = staticmethod(CalcAUTOCORR2D)
    # normalization
    # functions


class AUTOCORR3D_Featurizer(FixedSizeFeaturizer, MoleculeFeaturizer):
    # statics
    LENGTH = 80
    dtype = np.float32
    featurize = staticmethod(CalcAUTOCORR3D)
    # normalization
    # functions


class BCUT2D_Featurizer(FixedSizeFeaturizer, MoleculeFeaturizer):
    # statics
    LENGTH = 8
    dtype = np.float32
    featurize = staticmethod(BCUT2D)
    # normalization
    # functions


class CrippenDescriptors_Featurizer(FixedSizeFeaturizer, MoleculeFeaturizer):
    # statics
    LENGTH = 2
    dtype = np.float32
    featurize = staticmethod(CalcCrippenDescriptors)
    # normalization
    # functions


class EEMcharges_Featurizer(MoleculeFeaturizer):
    # statics
    dtype = np.float64

    featurize = staticmethod(CalcEEMcharges)
    # normalization
    # functions


class GETAWAY_Featurizer(FixedSizeFeaturizer, MoleculeFeaturizer):
    # statics
    LENGTH = 273
    dtype = np.float32
    featurize = staticmethod(CalcGETAWAY)
    # normalization
    # functions


class GetConnectivityInvariants_Featurizer(MoleculeFeaturizer):
    # statics
    dtype = np.int64

    featurize = staticmethod(GetConnectivityInvariants)
    # normalization
    # functions


class GetFeatureInvariants_Featurizer(MoleculeFeaturizer):
    # statics
    dtype = np.int64

    featurize = staticmethod(GetFeatureInvariants)
    # normalization
    # functions


class GetUSR_Featurizer(FixedSizeFeaturizer, MoleculeFeaturizer):
    # statics
    LENGTH = 12
    dtype = np.float32
    featurize = staticmethod(GetUSR)
    # normalization
    # functions


class GetUSRCAT_Featurizer(FixedSizeFeaturizer, MoleculeFeaturizer):
    # statics
    LENGTH = 60
    dtype = np.float32
    featurize = staticmethod(GetUSRCAT)
    # normalization
    # functions


class MORSE_Featurizer(FixedSizeFeaturizer, MoleculeFeaturizer):
    # statics
    LENGTH = 224
    dtype = np.float32
    featurize = staticmethod(CalcMORSE)
    # normalization
    # functions


class RDF_Featurizer(FixedSizeFeaturizer, MoleculeFeaturizer):
    # statics
    LENGTH = 210
    dtype = np.float32
    featurize = staticmethod(CalcRDF)
    # normalization
    # functions


class WHIM_Featurizer(FixedSizeFeaturizer, MoleculeFeaturizer):
    # statics
    LENGTH = 114
    dtype = np.float32
    featurize = staticmethod(CalcWHIM)
    # normalization
    # functions


molecule_AUTOCORR2D = AUTOCORR2D_Featurizer()
molecule_AUTOCORR3D = AUTOCORR3D_Featurizer()
molecule_BCUT2D = BCUT2D_Featurizer()
molecule_CrippenDescriptors = CrippenDescriptors_Featurizer()
molecule_EEMcharges = EEMcharges_Featurizer()
molecule_GETAWAY = GETAWAY_Featurizer()
molecule_GetConnectivityInvariants = GetConnectivityInvariants_Featurizer()
molecule_GetFeatureInvariants = GetFeatureInvariants_Featurizer()
molecule_GetUSR = GetUSR_Featurizer()
molecule_GetUSRCAT = GetUSRCAT_Featurizer()
molecule_MORSE = MORSE_Featurizer()
molecule_RDF = RDF_Featurizer()
molecule_WHIM = WHIM_Featurizer()

_available_featurizer = {
    "molecule_AUTOCORR2D": molecule_AUTOCORR2D,
    "molecule_AUTOCORR3D": molecule_AUTOCORR3D,
    "molecule_BCUT2D": molecule_BCUT2D,
    "molecule_CrippenDescriptors": molecule_CrippenDescriptors,
    "molecule_EEMcharges": molecule_EEMcharges,
    "molecule_GETAWAY": molecule_GETAWAY,
    "molecule_GetConnectivityInvariants": molecule_GetConnectivityInvariants,
    "molecule_GetFeatureInvariants": molecule_GetFeatureInvariants,
    "molecule_GetUSR": molecule_GetUSR,
    "molecule_GetUSRCAT": molecule_GetUSRCAT,
    "molecule_MORSE": molecule_MORSE,
    "molecule_RDF": molecule_RDF,
    "molecule_WHIM": molecule_WHIM,
}


def main():
    from rdkit import Chem

    testmol = Chem.MolFromSmiles("c1ccccc1")
    for k, f in _available_featurizer.items():
        print(k, f(testmol))


if __name__ == "__main__":
    main()
