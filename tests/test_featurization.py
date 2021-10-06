import logging
import sys
import unittest

import numpy as np

from molNet import MOLNET_LOGGER, ConformerError
from molNet.featurizer.molecule_featurizer import MolWtFeaturizer
from molNet.mol.molgraph import parallel_features_from_smiles, mol_graph_from_smiles
from molNet.utils.smiles.generator import generate_n_random_hetero_carbon_lattice

MOLNET_LOGGER.setLevel("DEBUG")


class FeatureTest(unittest.TestCase):
    def test_parallel_featurization(self):
        # SEED=263
        # for k in range(5):
        #    print("SEED",SEED)
        #    np.random.seed(SEED)
        #    random.seed(SEED)
        #    SEED=random.randint(0,1000)#901
        d=100
        d = np.array(
            [k for k in generate_n_random_hetero_carbon_lattice(n=d)]
        )
        f = MolWtFeaturizer()
        r = np.zeros((len(d), len(f))) * np.nan
        for i, _d in enumerate(d):
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
        assert len(feats) > 0.9*d, len(feats)
        assert np.allclose(feats, r), feats - r


logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
