import numpy as np
from rdkit.Chem import GetMolFrags

from molNet.featurizer._molecule_featurizer import MoleculeFeaturizer, SingleValueMoleculeFeaturizer


class ExtendMolnetFeaturizer(MoleculeFeaturizer):
    def featurize(self, mol):
        r = mol.molnet_features if hasattr(mol, "molnet_features") else []
        return np.array([r]).flatten()


extend_molnet_featurizer = ExtendMolnetFeaturizer(name="extend_molnet_featurizer")


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

_available_featurizer = {
    "molecule_num_fragments": molecule_num_fragments,
}


def get_available_featurizer():
    return _available_featurizer


__all__ = [
    "NumFragments_Featurizer",
    "molecule_num_fragments",

]


def main():
    from rdkit import Chem

    testmol = Chem.MolFromSmiles("c1ccccc1")

    for n, f in get_available_featurizer().items():
        print(f, f(testmol))


if __name__ == "__main__":
    main()
