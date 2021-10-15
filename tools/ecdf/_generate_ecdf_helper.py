from typing import List, Tuple

import numpy as np
from rdkit.Chem import MolFromSmiles, Mol, MolToSmiles

from molNet import ConformerError
from molNet.featurizer._molecule_featurizer import MoleculeFeaturizer

test_mol = MolFromSmiles("CCC")


def _single_call_parallel_featurize_molgraph(d: List[MoleculeFeaturizer]) -> List[np.ndarray]:
    feats = d
    r = []
    for f in feats:
        f.preferred_norm = None
        r.append(f(test_mol))
    return r


def _single_call_parallel_featurize_molfiles(d: Tuple[Mol, MoleculeFeaturizer]):
    feat = d[0][1]
    r = np.zeros((len(d), *feat(test_mol).shape)) * np.nan
    for i, data in enumerate(d):
        mol = data[0]
        feat = data[1]
        feat.preferred_norm = None
        try:
            print(MolToSmiles(mol))
            r[i] = feat(mol)
        except (ConformerError, ValueError, ZeroDivisionError):
            pass
    return r
