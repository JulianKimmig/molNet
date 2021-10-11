from molNet.featurizer._molecule_featurizer import (
    MoleculeFeaturizer,
    SingleValueMoleculeFeaturizer,
)
from molNet.featurizer.featurizer import FixedSizeFeaturizer
import numpy as np
from numpy import inf, nan
from rdkit.DataStructs.cDataStructs import ConvertToNumpyArray
from rdkit.Chem.rdmolops import (
    Get3DDistanceMatrix,
<<<<<<< HEAD
    GetDistanceMatrix,
    GetAdjacencyMatrix,
=======
    GetAdjacencyMatrix,
    GetDistanceMatrix,
>>>>>>> 3a4ef9b47d39261f7b23c48e5302154b6b288fc7
)


class Get3DDistanceMatrix_Featurizer(MoleculeFeaturizer):
    # statics
<<<<<<< HEAD
    dtype = np.float64
    # normalization
    # functions
=======
    dtype = np.int32
    # normalization
    # functions

    def featurize(self, mol):
        return GetAdjacencyMatrix(mol).flatten()

>>>>>>> 3a4ef9b47d39261f7b23c48e5302154b6b288fc7

    def featurize(self, mol):
        return Get3DDistanceMatrix(mol).flatten()


class GetAdjacencyMatrix_Featurizer(MoleculeFeaturizer):
    # statics
<<<<<<< HEAD
    dtype = np.int32
    # normalization
    # functions
=======
    dtype = np.float64
    # normalization
    # functions

    def featurize(self, mol):
        return GetDistanceMatrix(mol).flatten()

>>>>>>> 3a4ef9b47d39261f7b23c48e5302154b6b288fc7

    def featurize(self, mol):
        return GetAdjacencyMatrix(mol).flatten()


class GetDistanceMatrix_Featurizer(MoleculeFeaturizer):
    # statics
    dtype = np.float64
    # normalization
    # functions

    def featurize(self, mol):
<<<<<<< HEAD
        return GetDistanceMatrix(mol).flatten()


molecule_Get3DDistanceMatrix = Get3DDistanceMatrix_Featurizer()
molecule_GetAdjacencyMatrix = GetAdjacencyMatrix_Featurizer()
molecule_GetDistanceMatrix = GetDistanceMatrix_Featurizer()

_available_featurizer = {
    "molecule_Get3DDistanceMatrix": molecule_Get3DDistanceMatrix,
    "molecule_GetAdjacencyMatrix": molecule_GetAdjacencyMatrix,
    "molecule_GetDistanceMatrix": molecule_GetDistanceMatrix,
=======
        return Get3DDistanceMatrix(mol).flatten()


molecule_GetAdjacencyMatrix = GetAdjacencyMatrix_Featurizer()
molecule_GetDistanceMatrix = GetDistanceMatrix_Featurizer()
molecule_Get3DDistanceMatrix = Get3DDistanceMatrix_Featurizer()

_available_featurizer = {
    "molecule_GetAdjacencyMatrix": molecule_GetAdjacencyMatrix,
    "molecule_GetDistanceMatrix": molecule_GetDistanceMatrix,
    "molecule_Get3DDistanceMatrix": molecule_Get3DDistanceMatrix,
>>>>>>> 3a4ef9b47d39261f7b23c48e5302154b6b288fc7
}


def main():
    from rdkit import Chem
<<<<<<< HEAD

    testmol = Chem.MolFromSmiles("c1ccccc1")
    for k, f in _available_featurizer.items():
        print(k, f(testmol))

=======

    testmol = Chem.MolFromSmiles("c1ccccc1")
    for k, f in _available_featurizer.items():
        print(k)
        f(testmol)

>>>>>>> 3a4ef9b47d39261f7b23c48e5302154b6b288fc7

if __name__ == "__main__":
    main()
