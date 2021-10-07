from molNet.featurizer._molecule_featurizer import MoleculeFeaturizer,SingleValueMoleculeFeaturizer
from molNet.featurizer.featurizer import FixedSizeFeaturizer
import numpy as np
from rdkit.DataStructs.cDataStructs import ConvertToNumpyArray
from rdkit.Chem.rdMolDescriptors import (CalcMORSE,CalcAUTOCORR2D,CalcCrippenDescriptors,CalcRDF,BCUT2D,CalcWHIM,GetUSRCAT,GetFeatureInvariants,CalcAUTOCORR3D,GetConnectivityInvariants,GetUSR,CalcEEMcharges,CalcGETAWAY,)


class EEMcharges_Featurizer(MoleculeFeaturizer):
    dtype=np.float64
    
    featurize=staticmethod(CalcEEMcharges)
        

class BCUT2D_Featurizer(FixedSizeFeaturizer,MoleculeFeaturizer):
    LENGTH = 8
    dtype=np.float32
    featurize=staticmethod(BCUT2D)
        

class GetFeatureInvariants_Featurizer(MoleculeFeaturizer):
    dtype=np.int64
    
    featurize=staticmethod(GetFeatureInvariants)
        

class MORSE_Featurizer(FixedSizeFeaturizer,MoleculeFeaturizer):
    LENGTH = 224
    dtype=np.float32
    featurize=staticmethod(CalcMORSE)
        

class RDF_Featurizer(FixedSizeFeaturizer,MoleculeFeaturizer):
    LENGTH = 210
    dtype=np.float32
    featurize=staticmethod(CalcRDF)
        

class WHIM_Featurizer(FixedSizeFeaturizer,MoleculeFeaturizer):
    LENGTH = 114
    dtype=np.float32
    featurize=staticmethod(CalcWHIM)
        

class GetConnectivityInvariants_Featurizer(MoleculeFeaturizer):
    dtype=np.int64
    
    featurize=staticmethod(GetConnectivityInvariants)
        

class GetUSRCAT_Featurizer(FixedSizeFeaturizer,MoleculeFeaturizer):
    LENGTH = 60
    dtype=np.float32
    featurize=staticmethod(GetUSRCAT)
        

class GETAWAY_Featurizer(FixedSizeFeaturizer,MoleculeFeaturizer):
    LENGTH = 273
    dtype=np.float32
    featurize=staticmethod(CalcGETAWAY)
        

class AUTOCORR2D_Featurizer(FixedSizeFeaturizer,MoleculeFeaturizer):
    LENGTH = 192
    dtype=np.float32
    featurize=staticmethod(CalcAUTOCORR2D)
        

class CrippenDescriptors_Featurizer(FixedSizeFeaturizer,MoleculeFeaturizer):
    LENGTH = 2
    dtype=np.float32
    featurize=staticmethod(CalcCrippenDescriptors)
        

class AUTOCORR3D_Featurizer(FixedSizeFeaturizer,MoleculeFeaturizer):
    LENGTH = 80
    dtype=np.float32
    featurize=staticmethod(CalcAUTOCORR3D)
        

class GetUSR_Featurizer(FixedSizeFeaturizer,MoleculeFeaturizer):
    LENGTH = 12
    dtype=np.float32
    featurize=staticmethod(GetUSR)
        

molecule_EEMcharges=EEMcharges_Featurizer()
molecule_BCUT2D=BCUT2D_Featurizer()
molecule_GetFeatureInvariants=GetFeatureInvariants_Featurizer()
molecule_MORSE=MORSE_Featurizer()
molecule_RDF=RDF_Featurizer()
molecule_WHIM=WHIM_Featurizer()
molecule_GetConnectivityInvariants=GetConnectivityInvariants_Featurizer()
molecule_GetUSRCAT=GetUSRCAT_Featurizer()
molecule_GETAWAY=GETAWAY_Featurizer()
molecule_AUTOCORR2D=AUTOCORR2D_Featurizer()
molecule_CrippenDescriptors=CrippenDescriptors_Featurizer()
molecule_AUTOCORR3D=AUTOCORR3D_Featurizer()
molecule_GetUSR=GetUSR_Featurizer()

_available_featurizer={
'molecule_EEMcharges':molecule_EEMcharges,
'molecule_BCUT2D':molecule_BCUT2D,
'molecule_GetFeatureInvariants':molecule_GetFeatureInvariants,
'molecule_MORSE':molecule_MORSE,
'molecule_RDF':molecule_RDF,
'molecule_WHIM':molecule_WHIM,
'molecule_GetConnectivityInvariants':molecule_GetConnectivityInvariants,
'molecule_GetUSRCAT':molecule_GetUSRCAT,
'molecule_GETAWAY':molecule_GETAWAY,
'molecule_AUTOCORR2D':molecule_AUTOCORR2D,
'molecule_CrippenDescriptors':molecule_CrippenDescriptors,
'molecule_AUTOCORR3D':molecule_AUTOCORR3D,
'molecule_GetUSR':molecule_GetUSR
}




def main():
    from rdkit import Chem
    testmol=Chem.MolFromSmiles("c1ccccc1")
    for k,f in _available_featurizer.items():
        print(k)
        f(testmol)

if __name__=='__main__':
    main()
    