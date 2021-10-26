import numpy as np
from rdkit.Chem import (
    GetAdjacencyMatrix,
    GetDistanceMatrix,
    Get3DDistanceMatrix,
)

from molNet.featurizer._molecule_featurizer import (
    VarSizeMoleculeFeaturizer,
)


class AdjacencyMatrix_Featurizer(VarSizeMoleculeFeaturizer):
    # statics
    dtype = np.int32

    # normalization
    # functions

    def featurize(self, mol):
        return GetAdjacencyMatrix(mol).flatten()


class AdjacencyMatrix_Featurizer(VarSizeMoleculeFeaturizer):
    # statics
    dtype = np.int32

    # normalization
    # functions

    def featurize(self, mol):
        return GetAdjacencyMatrix(mol).flatten()


class DistanceMatrix_Featurizer(VarSizeMoleculeFeaturizer):
    # statics
    dtype = np.float64

    # normalization
    # functions

    def featurize(self, mol):
        return GetDistanceMatrix(mol).flatten()


class DistanceMatrix_Featurizer(VarSizeMoleculeFeaturizer):
    # statics
    dtype = np.float64

    # normalization
    # functions

    def featurize(self, mol):
        return GetDistanceMatrix(mol).flatten()


class Get3DDistanceMatrix_Featurizer(VarSizeMoleculeFeaturizer):
    # statics
    dtype = np.float64

    # normalization
    # functions

    def featurize(self, mol):
        return Get3DDistanceMatrix(mol).flatten()


class Get3DDistanceMatrix_Featurizer(VarSizeMoleculeFeaturizer):
    # statics
    dtype = np.float64

    # normalization
    # functions

    def featurize(self, mol):
        return Get3DDistanceMatrix(mol).flatten()


molecule_AdjacencyMatrix = AdjacencyMatrix_Featurizer()
molecule_AdjacencyMatrix = AdjacencyMatrix_Featurizer()
molecule_DistanceMatrix = DistanceMatrix_Featurizer()
molecule_DistanceMatrix = DistanceMatrix_Featurizer()
molecule_Get3DDistanceMatrix = Get3DDistanceMatrix_Featurizer()
molecule_Get3DDistanceMatrix = Get3DDistanceMatrix_Featurizer()

_available_featurizer = {
    "molecule_AdjacencyMatrix": molecule_AdjacencyMatrix,
    "molecule_AdjacencyMatrix": molecule_AdjacencyMatrix,
    "molecule_DistanceMatrix": molecule_DistanceMatrix,
    "molecule_DistanceMatrix": molecule_DistanceMatrix,
    "molecule_Get3DDistanceMatrix": molecule_Get3DDistanceMatrix,
    "molecule_Get3DDistanceMatrix": molecule_Get3DDistanceMatrix,
}


def get_available_featurizer():
    return _available_featurizer


__all__ = [
    "AdjacencyMatrix_Featurizer",
    "AdjacencyMatrix_Featurizer",
    "DistanceMatrix_Featurizer",
    "DistanceMatrix_Featurizer",
    "Get3DDistanceMatrix_Featurizer",
    "Get3DDistanceMatrix_Featurizer",
    "molecule_AdjacencyMatrix",
    "molecule_AdjacencyMatrix",
    "molecule_DistanceMatrix",
    "molecule_DistanceMatrix",
    "molecule_Get3DDistanceMatrix",
    "molecule_Get3DDistanceMatrix",
]


def main():
    from rdkit import Chem

    testmol = Chem.MolFromSmiles("c1ccccc1")
    for n, f in get_available_featurizer().items():
        print(n, f(testmol))


if __name__ == "__main__":
    main()
