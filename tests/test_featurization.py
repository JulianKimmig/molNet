import logging
import sys
import unittest

import numpy as np
from rdkit.Chem.Lipinski import HeavyAtomCount
from tqdm import tqdm

from molNet import MOLNET_LOGGER, ConformerError
from molNet.featurizer._autogen_molecule_featurizer import MolWtFeaturizer
from molNet.mol.molgraph import parallel_features_from_smiles, mol_graph_from_smiles
from molNet.utils.mol.generator import generate_random_carbon_lattice
from molNet.utils.smiles.generator import generate_n_random_hetero_carbon_lattice

MOLNET_LOGGER.setLevel("DEBUG")


class FeatureTest(unittest.TestCase):
    def test_basic_featurizer(self):
        from molNet.featurizer.normalization import NormalizationClass

        norm = NormalizationClass(
            min_max_norm_parameter=(-1, 1), preferred_normalization="min_max"
        )
        res = norm.normalize(np.arange(-4, 4))
        assert np.allclose(res, np.array([0, 0, 0, 0, 0.5, 1, 1, 1]))

        class TestNorm(NormalizationClass):
            linear_norm_parameter = (10, 0)
            preferred_normalization = "min_max"

        from molNet.featurizer.normalization import NormalizationException

        self.assertRaises(NormalizationException, TestNorm)

        class TestNorm(NormalizationClass):
            linear_norm_parameter = (10, 0)
            preferred_normalization = "linear"

        norm = TestNorm()
        res = norm.normalize(np.arange(-4, 4))
        assert np.allclose(res, np.arange(-4, 4) * 10), res

        from molNet.featurizer.featurizer import Featurizer

        class TestF(Featurizer):
            dtype = np.uint32

            def featurize(self, ob):
                return ob.shape

        f = TestF()
        res = f(np.zeros((10, 2, 1)))
        assert np.allclose(res, np.array((10, 2, 1)))

        from molNet.featurizer._molecule_featurizer import SingleValueMoleculeFeaturizer

        class TSVMF(SingleValueMoleculeFeaturizer):
            LENGTH = 1

            featurize = staticmethod(HeavyAtomCount)

        c = generate_random_carbon_lattice(n=4)
        assert TSVMF()(c) == 4

    def test_parallel_featurization(self):
        # SEED=263
        # for k in range(5):
        #    print("SEED",SEED)
        #    np.random.seed(SEED)
        #    random.seed(SEED)
        #    SEED=random.randint(0,1000)#901
        l = 100
        d = np.array(
            [k for k in generate_n_random_hetero_carbon_lattice(n=l, max_c=10)]
        )
        f = MolWtFeaturizer()
        r = np.zeros((len(d), len(f))) * np.nan
        for i, _d in tqdm(enumerate(d), total=len(d)):
            try:
                mg = mol_graph_from_smiles(_d)
                mg.featurize_mol(f, name="para_feats")
                r[i] = mg.as_arrays()["graph_features"]["para_feats"]
            except ConformerError:
                pass
        r = r.flatten()
        d = d[~np.isnan(r)]
        r = r[~np.isnan(r)]
        feats = parallel_features_from_smiles(d, MolWtFeaturizer)

        r = r[~np.isnan(feats)]
        feats = feats[~np.isnan(feats)]
        assert len(feats) > l * 0.9, len(feats)
        assert np.allclose(feats, r), feats - r


logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
