import numpy as np
from rdkit.Chem import (GetMolFrags, Get3DDistanceMatrix, GetDistanceMatrix, GetAdjacencyMatrix)
from rdkit.Chem.AllChem import (CalcCoulombMat, CalcGETAWAY, CalcAUTOCORR2D, CalcMORSE, CalcAUTOCORR3D,
                                GetConnectivityInvariants, BCUT2D, GetUSRCAT, GetFeatureInvariants, GetErGFingerprint,
                                CalcRDF, GetUSR, CalcEEMcharges, CalcCrippenDescriptors, CalcWHIM)
from rdkit.Chem.EState import (EStateIndices)
from rdkit.Chem.QED import (properties)

from molNet.featurizer._molecule_featurizer import (FixedSizeMoleculeFeaturizer, VarSizeMoleculeFeaturizer)
from molNet.featurizer.molecule_featurizer import prepare_mol_for_featurization


class Molecule_EEMcharges_Featurizer(VarSizeMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.AllChem.CalcEEMcharges
    dtype = np.float32
    featurize = staticmethod(CalcEEMcharges)


class Molecule_BCUT2D_Featurizer(FixedSizeMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.AllChem.BCUT2D
    dtype = np.float32
    featurize = staticmethod(BCUT2D)
    LENGTH = 8


class Molecule_MORSE_Featurizer(FixedSizeMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.AllChem.CalcMORSE
    dtype = np.float32
    featurize = staticmethod(CalcMORSE)
    LENGTH = 224


class Molecule_CoulombMat_Featurizer(VarSizeMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.AllChem.CalcCoulombMat
    dtype = np.float32
    featurize = staticmethod(CalcCoulombMat)


class Molecule_FeatureInvariants_Featurizer(VarSizeMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.AllChem.GetFeatureInvariants
    dtype = np.int32
    featurize = staticmethod(GetFeatureInvariants)


class Molecule_AUTOCORR2D_Featurizer(FixedSizeMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.AllChem.CalcAUTOCORR2D
    dtype = np.float32
    featurize = staticmethod(CalcAUTOCORR2D)
    LENGTH = 192


class Molecule_WHIM_Featurizer(FixedSizeMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.AllChem.CalcWHIM
    dtype = np.float32
    featurize = staticmethod(CalcWHIM)
    LENGTH = 114


class Molecule_ConnectivityInvariants_Featurizer(VarSizeMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.AllChem.GetConnectivityInvariants
    dtype = np.int64
    featurize = staticmethod(GetConnectivityInvariants)


class Molecule_USRCAT_Featurizer(FixedSizeMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.AllChem.GetUSRCAT
    dtype = np.float32
    featurize = staticmethod(GetUSRCAT)
    LENGTH = 60


class Molecule_ErGFingerprint_Featurizer(FixedSizeMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.AllChem.GetErGFingerprint
    dtype = np.float32
    featurize = staticmethod(GetErGFingerprint)
    LENGTH = 315


class Molecule_properties_Featurizer(FixedSizeMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.QED.properties
    dtype = np.float32
    featurize = staticmethod(properties)
    LENGTH = 8


class Molecule_MolFrags_Featurizer(FixedSizeMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.GetMolFrags
    dtype = np.int32
    featurize = staticmethod(GetMolFrags)
    LENGTH = 1


class Molecule_EStateIndices_Featurizer(VarSizeMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.EState.EStateIndices
    dtype = np.float32
    featurize = staticmethod(EStateIndices)


class Molecule_AdjacencyMatrix_Featurizer(VarSizeMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.GetAdjacencyMatrix
    dtype = np.int32
    featurize = staticmethod(GetAdjacencyMatrix)


class Molecule_RDF_Featurizer(FixedSizeMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.AllChem.CalcRDF
    dtype = np.float32
    featurize = staticmethod(CalcRDF)
    LENGTH = 210


class Molecule_Get3DDistanceMatrix_Featurizer(VarSizeMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.Get3DDistanceMatrix
    dtype = np.float32
    featurize = staticmethod(Get3DDistanceMatrix)


class Molecule_AUTOCORR3D_Featurizer(FixedSizeMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.AllChem.CalcAUTOCORR3D
    dtype = np.float32
    featurize = staticmethod(CalcAUTOCORR3D)
    LENGTH = 80


class Molecule_USR_Featurizer(FixedSizeMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.AllChem.GetUSR
    dtype = np.float32
    featurize = staticmethod(GetUSR)
    LENGTH = 12


class Molecule_DistanceMatrix_Featurizer(VarSizeMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.GetDistanceMatrix
    dtype = np.float32
    featurize = staticmethod(GetDistanceMatrix)


class Molecule_GETAWAY_Featurizer(FixedSizeMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.AllChem.CalcGETAWAY
    dtype = np.float32
    featurize = staticmethod(CalcGETAWAY)
    LENGTH = 273


class Molecule_CrippenDescriptors_Featurizer(FixedSizeMoleculeFeaturizer):
    # _rdfunc=rdkit.Chem.AllChem.CalcCrippenDescriptors
    dtype = np.float32
    featurize = staticmethod(CalcCrippenDescriptors)
    LENGTH = 2


molecule_EEMcharges_featurizer = Molecule_EEMcharges_Featurizer()
molecule_BCUT2D_featurizer = Molecule_BCUT2D_Featurizer()
molecule_MORSE_featurizer = Molecule_MORSE_Featurizer()
molecule_CoulombMat_featurizer = Molecule_CoulombMat_Featurizer()
molecule_FeatureInvariants_featurizer = Molecule_FeatureInvariants_Featurizer()
molecule_AUTOCORR2D_featurizer = Molecule_AUTOCORR2D_Featurizer()
molecule_WHIM_featurizer = Molecule_WHIM_Featurizer()
molecule_ConnectivityInvariants_featurizer = Molecule_ConnectivityInvariants_Featurizer()
molecule_USRCAT_featurizer = Molecule_USRCAT_Featurizer()
molecule_ErGFingerprint_featurizer = Molecule_ErGFingerprint_Featurizer()
molecule_properties_featurizer = Molecule_properties_Featurizer()
molecule_MolFrags_featurizer = Molecule_MolFrags_Featurizer()
molecule_EStateIndices_featurizer = Molecule_EStateIndices_Featurizer()
molecule_AdjacencyMatrix_featurizer = Molecule_AdjacencyMatrix_Featurizer()
molecule_RDF_featurizer = Molecule_RDF_Featurizer()
molecule_Get3DDistanceMatrix_featurizer = Molecule_Get3DDistanceMatrix_Featurizer()
molecule_AUTOCORR3D_featurizer = Molecule_AUTOCORR3D_Featurizer()
molecule_USR_featurizer = Molecule_USR_Featurizer()
molecule_DistanceMatrix_featurizer = Molecule_DistanceMatrix_Featurizer()
molecule_GETAWAY_featurizer = Molecule_GETAWAY_Featurizer()
molecule_CrippenDescriptors_featurizer = Molecule_CrippenDescriptors_Featurizer()
_available_featurizer = {
    'molecule_EEMcharges_featurizer': molecule_EEMcharges_featurizer,
    'molecule_BCUT2D_featurizer': molecule_BCUT2D_featurizer,
    'molecule_MORSE_featurizer': molecule_MORSE_featurizer,
    'molecule_CoulombMat_featurizer': molecule_CoulombMat_featurizer,
    'molecule_FeatureInvariants_featurizer': molecule_FeatureInvariants_featurizer,
    'molecule_AUTOCORR2D_featurizer': molecule_AUTOCORR2D_featurizer,
    'molecule_WHIM_featurizer': molecule_WHIM_featurizer,
    'molecule_ConnectivityInvariants_featurizer': molecule_ConnectivityInvariants_featurizer,
    'molecule_USRCAT_featurizer': molecule_USRCAT_featurizer,
    'molecule_ErGFingerprint_featurizer': molecule_ErGFingerprint_featurizer,
    'molecule_properties_featurizer': molecule_properties_featurizer,
    'molecule_MolFrags_featurizer': molecule_MolFrags_featurizer,
    'molecule_EStateIndices_featurizer': molecule_EStateIndices_featurizer,
    'molecule_AdjacencyMatrix_featurizer': molecule_AdjacencyMatrix_featurizer,
    'molecule_RDF_featurizer': molecule_RDF_featurizer,
    'molecule_Get3DDistanceMatrix_featurizer': molecule_Get3DDistanceMatrix_featurizer,
    'molecule_AUTOCORR3D_featurizer': molecule_AUTOCORR3D_featurizer,
    'molecule_USR_featurizer': molecule_USR_featurizer,
    'molecule_DistanceMatrix_featurizer': molecule_DistanceMatrix_featurizer,
    'molecule_GETAWAY_featurizer': molecule_GETAWAY_featurizer,
    'molecule_CrippenDescriptors_featurizer': molecule_CrippenDescriptors_featurizer,
}
__all__ = [
    'Molecule_EEMcharges_Featurizer',
    'molecule_EEMcharges_featurizer',
    'Molecule_BCUT2D_Featurizer',
    'molecule_BCUT2D_featurizer',
    'Molecule_MORSE_Featurizer',
    'molecule_MORSE_featurizer',
    'Molecule_CoulombMat_Featurizer',
    'molecule_CoulombMat_featurizer',
    'Molecule_FeatureInvariants_Featurizer',
    'molecule_FeatureInvariants_featurizer',
    'Molecule_AUTOCORR2D_Featurizer',
    'molecule_AUTOCORR2D_featurizer',
    'Molecule_WHIM_Featurizer',
    'molecule_WHIM_featurizer',
    'Molecule_ConnectivityInvariants_Featurizer',
    'molecule_ConnectivityInvariants_featurizer',
    'Molecule_USRCAT_Featurizer',
    'molecule_USRCAT_featurizer',
    'Molecule_ErGFingerprint_Featurizer',
    'molecule_ErGFingerprint_featurizer',
    'Molecule_properties_Featurizer',
    'molecule_properties_featurizer',
    'Molecule_MolFrags_Featurizer',
    'molecule_MolFrags_featurizer',
    'Molecule_EStateIndices_Featurizer',
    'molecule_EStateIndices_featurizer',
    'Molecule_AdjacencyMatrix_Featurizer',
    'molecule_AdjacencyMatrix_featurizer',
    'Molecule_RDF_Featurizer',
    'molecule_RDF_featurizer',
    'Molecule_Get3DDistanceMatrix_Featurizer',
    'molecule_Get3DDistanceMatrix_featurizer',
    'Molecule_AUTOCORR3D_Featurizer',
    'molecule_AUTOCORR3D_featurizer',
    'Molecule_USR_Featurizer',
    'molecule_USR_featurizer',
    'Molecule_DistanceMatrix_Featurizer',
    'molecule_DistanceMatrix_featurizer',
    'Molecule_GETAWAY_Featurizer',
    'molecule_GETAWAY_featurizer',
    'Molecule_CrippenDescriptors_Featurizer',
    'molecule_CrippenDescriptors_featurizer',
]


def get_available_featurizer():
    return _available_featurizer


def main():
    from rdkit import Chem
    testdata = prepare_mol_for_featurization(Chem.MolFromSmiles('c1ccccc1'))
    for n, f in get_available_featurizer().items():
        print(n, end=' ')
        print(f(testdata))
    print(len(get_available_featurizer()))


if __name__ == '__main__':
    main()
