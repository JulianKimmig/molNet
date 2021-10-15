import numpy as np
from rdkit.Chem.rdmolops import (
    GetAdjacencyMatrix,
    Get3DDistanceMatrix,
    GetDistanceMatrix,
)

from molNet.featurizer._molecule_featurizer import (
    VarSizeMoleculeFeaturizer,
)


class Get3DDistanceMatrix_Featurizer(VarSizeMoleculeFeaturizer):
    # statics
    dtype = np.float64

    # normalization
    # functions

    def featurize(self, mol):
        return Get3DDistanceMatrix(mol).flatten()


class GetAdjacencyMatrix_Featurizer(VarSizeMoleculeFeaturizer):
    # statics
    dtype = np.int32
    # normalization
    # functions

    def featurize(self, mol):
        return GetAdjacencyMatrix(mol).flatten()


class GetDistanceMatrix_Featurizer(VarSizeMoleculeFeaturizer):
    # statics
    dtype = np.float64
    # normalization
    # functions

    def featurize(self, mol):
        return GetDistanceMatrix(mol).flatten()


molecule_Get3DDistanceMatrix = Get3DDistanceMatrix_Featurizer()
molecule_GetAdjacencyMatrix = GetAdjacencyMatrix_Featurizer()
molecule_GetDistanceMatrix = GetDistanceMatrix_Featurizer()

_available_featurizer = [
    molecule_Get3DDistanceMatrix,
    molecule_GetAdjacencyMatrix,
    molecule_GetDistanceMatrix,
]


def main():
    from rdkit import Chem

    testmol = Chem.MolFromSmiles("c1ccccc1")
    for f in _available_featurizer.items():
        print(f, f(testmol))


if __name__ == "__main__":
    main()
