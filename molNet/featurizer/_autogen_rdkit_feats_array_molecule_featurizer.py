import numpy as np
from molNet.featurizer._molecule_featurizer import (
    FixedSizeMoleculeFeaturizer,
    VarSizeMoleculeFeaturizer,
)
from rdkit.Chem.AllChem import (
    GetFeatureInvariants,
    GetUSR,
    GetUSRCAT,
    GetConnectivityInvariants,
    CalcEEMcharges,
    CalcCoulombMat,
    CalcAUTOCORR3D,
    GetErGFingerprint,
    Get3DDistanceMatrix,
    GetAdjacencyMatrix,
    CalcCrippenDescriptors,
    CalcMORSE,
    CalcAUTOCORR2D,
    CalcRDF,
    BCUT2D,
    CalcGETAWAY,
    CalcWHIM,
)
from rdkit.Chem import Get3DDistanceMatrix, GetAdjacencyMatrix, GetDistanceMatrix
from rdkit.Chem.EState import EStateIndices
from rdkit.Chem.QED import properties
from rdkit.Chem.rdmolops import GetDistanceMatrix


class Molecule_AllChem_AUTOCORR2D_Featurizer(FixedSizeMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.AllChem.CalcAUTOCORR2D
    dtype = np.float32
    featurize = staticmethod(CalcAUTOCORR2D)
    LENGTH = 192


class Molecule_AllChem_AUTOCORR3D_Featurizer(FixedSizeMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.AllChem.CalcAUTOCORR3D
    dtype = np.float32
    featurize = staticmethod(CalcAUTOCORR3D)
    LENGTH = 80


class Molecule_AllChem_AdjacencyMatrix_Featurizer(VarSizeMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.AllChem.GetAdjacencyMatrix
    dtype = np.int32
    featurize = staticmethod(GetAdjacencyMatrix)


class Molecule_AllChem_BCUT2D_Featurizer(FixedSizeMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.AllChem.BCUT2D
    dtype = np.float32
    featurize = staticmethod(BCUT2D)
    LENGTH = 8


class Molecule_AllChem_ConnectivityInvariants_Featurizer(VarSizeMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.AllChem.GetConnectivityInvariants
    dtype = np.int64
    featurize = staticmethod(GetConnectivityInvariants)


class Molecule_AllChem_CoulombMat_Featurizer(VarSizeMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.AllChem.CalcCoulombMat
    dtype = np.float32
    featurize = staticmethod(CalcCoulombMat)


class Molecule_AllChem_CrippenDescriptors_Featurizer(FixedSizeMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.AllChem.CalcCrippenDescriptors
    dtype = np.float32
    featurize = staticmethod(CalcCrippenDescriptors)
    LENGTH = 2


class Molecule_AllChem_EEMcharges_Featurizer(VarSizeMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.AllChem.CalcEEMcharges
    dtype = np.float32
    featurize = staticmethod(CalcEEMcharges)


class Molecule_AllChem_ErGFingerprint_Featurizer(FixedSizeMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.AllChem.GetErGFingerprint
    dtype = np.float32
    featurize = staticmethod(GetErGFingerprint)
    LENGTH = 315


class Molecule_AllChem_FeatureInvariants_Featurizer(VarSizeMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.AllChem.GetFeatureInvariants
    dtype = np.int32
    featurize = staticmethod(GetFeatureInvariants)


class Molecule_AllChem_GETAWAY_Featurizer(FixedSizeMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.AllChem.CalcGETAWAY
    dtype = np.float32
    featurize = staticmethod(CalcGETAWAY)
    LENGTH = 273


class Molecule_AllChem_Get3DDistanceMatrix_Featurizer(VarSizeMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.AllChem.Get3DDistanceMatrix
    dtype = np.float32
    featurize = staticmethod(Get3DDistanceMatrix)


class Molecule_AllChem_MORSE_Featurizer(FixedSizeMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.AllChem.CalcMORSE
    dtype = np.float32
    featurize = staticmethod(CalcMORSE)
    LENGTH = 224


class Molecule_AllChem_RDF_Featurizer(FixedSizeMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.AllChem.CalcRDF
    dtype = np.float32
    featurize = staticmethod(CalcRDF)
    LENGTH = 210


class Molecule_AllChem_USRCAT_Featurizer(FixedSizeMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.AllChem.GetUSRCAT
    dtype = np.float32
    featurize = staticmethod(GetUSRCAT)
    LENGTH = 60


class Molecule_AllChem_USR_Featurizer(FixedSizeMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.AllChem.GetUSR
    dtype = np.float32
    featurize = staticmethod(GetUSR)
    LENGTH = 12


class Molecule_AllChem_WHIM_Featurizer(FixedSizeMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.AllChem.CalcWHIM
    dtype = np.float32
    featurize = staticmethod(CalcWHIM)
    LENGTH = 114


class Molecule_Chem_AdjacencyMatrix_Featurizer(VarSizeMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.GetAdjacencyMatrix
    dtype = np.int32
    featurize = staticmethod(GetAdjacencyMatrix)


class Molecule_Chem_DistanceMatrix_Featurizer(VarSizeMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.GetDistanceMatrix
    dtype = np.float32
    featurize = staticmethod(GetDistanceMatrix)


class Molecule_Chem_Get3DDistanceMatrix_Featurizer(VarSizeMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Get3DDistanceMatrix
    dtype = np.float32
    featurize = staticmethod(Get3DDistanceMatrix)


class Molecule_EState_EStateIndices_Featurizer(VarSizeMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.EState.EStateIndices
    dtype = np.float32
    featurize = staticmethod(EStateIndices)


class Molecule_QED_properties_Featurizer(FixedSizeMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.QED.properties
    dtype = np.float32
    featurize = staticmethod(properties)
    LENGTH = 8


class Molecule_rdmolops_DistanceMatrix_Featurizer(VarSizeMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.rdmolops.GetDistanceMatrix
    dtype = np.float32
    featurize = staticmethod(GetDistanceMatrix)


molecule_AllChem_AUTOCORR2D_featurizer = Molecule_AllChem_AUTOCORR2D_Featurizer()
molecule_AllChem_AUTOCORR3D_featurizer = Molecule_AllChem_AUTOCORR3D_Featurizer()
molecule_AllChem_AdjacencyMatrix_featurizer = (
    Molecule_AllChem_AdjacencyMatrix_Featurizer()
)
molecule_AllChem_BCUT2D_featurizer = Molecule_AllChem_BCUT2D_Featurizer()
molecule_AllChem_ConnectivityInvariants_featurizer = (
    Molecule_AllChem_ConnectivityInvariants_Featurizer()
)
molecule_AllChem_CoulombMat_featurizer = Molecule_AllChem_CoulombMat_Featurizer()
molecule_AllChem_CrippenDescriptors_featurizer = (
    Molecule_AllChem_CrippenDescriptors_Featurizer()
)
molecule_AllChem_EEMcharges_featurizer = Molecule_AllChem_EEMcharges_Featurizer()
molecule_AllChem_ErGFingerprint_featurizer = (
    Molecule_AllChem_ErGFingerprint_Featurizer()
)
molecule_AllChem_FeatureInvariants_featurizer = (
    Molecule_AllChem_FeatureInvariants_Featurizer()
)
molecule_AllChem_GETAWAY_featurizer = Molecule_AllChem_GETAWAY_Featurizer()
molecule_AllChem_Get3DDistanceMatrix_featurizer = (
    Molecule_AllChem_Get3DDistanceMatrix_Featurizer()
)
molecule_AllChem_MORSE_featurizer = Molecule_AllChem_MORSE_Featurizer()
molecule_AllChem_RDF_featurizer = Molecule_AllChem_RDF_Featurizer()
molecule_AllChem_USRCAT_featurizer = Molecule_AllChem_USRCAT_Featurizer()
molecule_AllChem_USR_featurizer = Molecule_AllChem_USR_Featurizer()
molecule_AllChem_WHIM_featurizer = Molecule_AllChem_WHIM_Featurizer()
molecule_Chem_AdjacencyMatrix_featurizer = Molecule_Chem_AdjacencyMatrix_Featurizer()
molecule_Chem_DistanceMatrix_featurizer = Molecule_Chem_DistanceMatrix_Featurizer()
molecule_Chem_Get3DDistanceMatrix_featurizer = (
    Molecule_Chem_Get3DDistanceMatrix_Featurizer()
)
molecule_EState_EStateIndices_featurizer = Molecule_EState_EStateIndices_Featurizer()
molecule_QED_properties_featurizer = Molecule_QED_properties_Featurizer()
molecule_rdmolops_DistanceMatrix_featurizer = (
    Molecule_rdmolops_DistanceMatrix_Featurizer()
)
_available_featurizer = {
    "molecule_AllChem_AUTOCORR2D_featurizer": molecule_AllChem_AUTOCORR2D_featurizer,
    "molecule_AllChem_AUTOCORR3D_featurizer": molecule_AllChem_AUTOCORR3D_featurizer,
    "molecule_AllChem_AdjacencyMatrix_featurizer": molecule_AllChem_AdjacencyMatrix_featurizer,
    "molecule_AllChem_BCUT2D_featurizer": molecule_AllChem_BCUT2D_featurizer,
    "molecule_AllChem_ConnectivityInvariants_featurizer": molecule_AllChem_ConnectivityInvariants_featurizer,
    "molecule_AllChem_CoulombMat_featurizer": molecule_AllChem_CoulombMat_featurizer,
    "molecule_AllChem_CrippenDescriptors_featurizer": molecule_AllChem_CrippenDescriptors_featurizer,
    "molecule_AllChem_EEMcharges_featurizer": molecule_AllChem_EEMcharges_featurizer,
    "molecule_AllChem_ErGFingerprint_featurizer": molecule_AllChem_ErGFingerprint_featurizer,
    "molecule_AllChem_FeatureInvariants_featurizer": molecule_AllChem_FeatureInvariants_featurizer,
    "molecule_AllChem_GETAWAY_featurizer": molecule_AllChem_GETAWAY_featurizer,
    "molecule_AllChem_Get3DDistanceMatrix_featurizer": molecule_AllChem_Get3DDistanceMatrix_featurizer,
    "molecule_AllChem_MORSE_featurizer": molecule_AllChem_MORSE_featurizer,
    "molecule_AllChem_RDF_featurizer": molecule_AllChem_RDF_featurizer,
    "molecule_AllChem_USRCAT_featurizer": molecule_AllChem_USRCAT_featurizer,
    "molecule_AllChem_USR_featurizer": molecule_AllChem_USR_featurizer,
    "molecule_AllChem_WHIM_featurizer": molecule_AllChem_WHIM_featurizer,
    "molecule_Chem_AdjacencyMatrix_featurizer": molecule_Chem_AdjacencyMatrix_featurizer,
    "molecule_Chem_DistanceMatrix_featurizer": molecule_Chem_DistanceMatrix_featurizer,
    "molecule_Chem_Get3DDistanceMatrix_featurizer": molecule_Chem_Get3DDistanceMatrix_featurizer,
    "molecule_EState_EStateIndices_featurizer": molecule_EState_EStateIndices_featurizer,
    "molecule_QED_properties_featurizer": molecule_QED_properties_featurizer,
    "molecule_rdmolops_DistanceMatrix_featurizer": molecule_rdmolops_DistanceMatrix_featurizer,
}
__all__ = [
    "Molecule_AllChem_AUTOCORR2D_Featurizer",
    "molecule_AllChem_AUTOCORR2D_featurizer",
    "Molecule_AllChem_AUTOCORR3D_Featurizer",
    "molecule_AllChem_AUTOCORR3D_featurizer",
    "Molecule_AllChem_AdjacencyMatrix_Featurizer",
    "molecule_AllChem_AdjacencyMatrix_featurizer",
    "Molecule_AllChem_BCUT2D_Featurizer",
    "molecule_AllChem_BCUT2D_featurizer",
    "Molecule_AllChem_ConnectivityInvariants_Featurizer",
    "molecule_AllChem_ConnectivityInvariants_featurizer",
    "Molecule_AllChem_CoulombMat_Featurizer",
    "molecule_AllChem_CoulombMat_featurizer",
    "Molecule_AllChem_CrippenDescriptors_Featurizer",
    "molecule_AllChem_CrippenDescriptors_featurizer",
    "Molecule_AllChem_EEMcharges_Featurizer",
    "molecule_AllChem_EEMcharges_featurizer",
    "Molecule_AllChem_ErGFingerprint_Featurizer",
    "molecule_AllChem_ErGFingerprint_featurizer",
    "Molecule_AllChem_FeatureInvariants_Featurizer",
    "molecule_AllChem_FeatureInvariants_featurizer",
    "Molecule_AllChem_GETAWAY_Featurizer",
    "molecule_AllChem_GETAWAY_featurizer",
    "Molecule_AllChem_Get3DDistanceMatrix_Featurizer",
    "molecule_AllChem_Get3DDistanceMatrix_featurizer",
    "Molecule_AllChem_MORSE_Featurizer",
    "molecule_AllChem_MORSE_featurizer",
    "Molecule_AllChem_RDF_Featurizer",
    "molecule_AllChem_RDF_featurizer",
    "Molecule_AllChem_USRCAT_Featurizer",
    "molecule_AllChem_USRCAT_featurizer",
    "Molecule_AllChem_USR_Featurizer",
    "molecule_AllChem_USR_featurizer",
    "Molecule_AllChem_WHIM_Featurizer",
    "molecule_AllChem_WHIM_featurizer",
    "Molecule_Chem_AdjacencyMatrix_Featurizer",
    "molecule_Chem_AdjacencyMatrix_featurizer",
    "Molecule_Chem_DistanceMatrix_Featurizer",
    "molecule_Chem_DistanceMatrix_featurizer",
    "Molecule_Chem_Get3DDistanceMatrix_Featurizer",
    "molecule_Chem_Get3DDistanceMatrix_featurizer",
    "Molecule_EState_EStateIndices_Featurizer",
    "molecule_EState_EStateIndices_featurizer",
    "Molecule_QED_properties_Featurizer",
    "molecule_QED_properties_featurizer",
    "Molecule_rdmolops_DistanceMatrix_Featurizer",
    "molecule_rdmolops_DistanceMatrix_featurizer",
]


def get_available_featurizer():
    return _available_featurizer


def main():
    from rdkit import Chem
    from molNet.featurizer.molecule_featurizer import prepare_mol_for_featurization

    testdata = prepare_mol_for_featurization(Chem.MolFromSmiles("c1ccccc1"))
    for n, f in get_available_featurizer().items():
        print(n, end=" ")
        print(f(testdata))
    print(len(get_available_featurizer()))


if __name__ == "__main__":
    main()
