from molNet.featurizer._molecule_featurizer import MoleculeFeaturizer,SingleValueMoleculeFeaturizer
from molNet.featurizer.featurizer import FixedSizeFeaturizer
import numpy as np
from rdkit.DataStructs.cDataStructs import ConvertToNumpyArray
from rdkit.Chem.rdmolops import (GetDistanceMatrix,GetAdjacencyMatrix,Get3DDistanceMatrix,)


class GetDistanceMatrix_Featurizer(MoleculeFeaturizer):
    dtype=np.float64
    
    def featurize(self,mol):
        return GetDistanceMatrix(mol).flatten()
        

class GetAdjacencyMatrix_Featurizer(MoleculeFeaturizer):
    dtype=np.int32
    
    def featurize(self,mol):
        return GetAdjacencyMatrix(mol).flatten()
        

class Get3DDistanceMatrix_Featurizer(MoleculeFeaturizer):
    dtype=np.float64
    
    def featurize(self,mol):
        return Get3DDistanceMatrix(mol).flatten()
        

molecule_GetDistanceMatrix=GetDistanceMatrix_Featurizer()
molecule_GetAdjacencyMatrix=GetAdjacencyMatrix_Featurizer()
molecule_Get3DDistanceMatrix=Get3DDistanceMatrix_Featurizer()

_available_featurizer={
'molecule_GetDistanceMatrix':molecule_GetDistanceMatrix,
'molecule_GetAdjacencyMatrix':molecule_GetAdjacencyMatrix,
'molecule_Get3DDistanceMatrix':molecule_Get3DDistanceMatrix
}




def main():
    from rdkit import Chem
    testmol=Chem.MolFromSmiles("c1ccccc1")
    for k,f in _available_featurizer.items():
        print(k)
        f(testmol)

if __name__=='__main__':
    main()
    