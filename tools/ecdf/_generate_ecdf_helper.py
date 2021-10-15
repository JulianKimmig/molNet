import inspect
import json
import os
from typing import List, Tuple, Dict

import numpy as np
from rdkit.Chem import MolFromSmiles, Mol, MolToSmiles

from molNet import ConformerError
from molNet.featurizer._molecule_featurizer import MoleculeFeaturizer, VarSizeMoleculeFeaturizer
from molNet.utils.parallelization.multiprocessing import parallelize

test_mol = MolFromSmiles("CCC")
basedir_molecule_featurizer = os.path.dirname(inspect.getfile(MoleculeFeaturizer))


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


def get_molecule_featurizer(ignored_names=None, ignore_var_size=True):
    if ignored_names is None:
        ignored_names = []
    import molNet.featurizer.molecule_featurizer as mf

    molfeats = mf._available_featurizer
    print(f"found {len(molfeats)} molecule featurizer")

    # bool is already between 0 and 1
    molfeats = [f for f in molfeats if f.dtype != bool]
    print(f"{len(molfeats)} remain after removal of bool types")
    molfeats = [f for f in molfeats if str(f) not in ignored_names]
    if ignore_var_size:
        molfeats = [f for f in molfeats if not isinstance(f, VarSizeMoleculeFeaturizer)]
    print(f"{len(molfeats)} remain after removal of ignored")

    generated_test_feats = parallelize(
        _single_call_parallel_featurize_molgraph,
        molfeats,
        cores="all-1",
        progess_bar=True,
        progress_bar_kwargs=dict(unit=" feats"),
    )

    molfeats = [molfeats[i] for i in range(len(molfeats)) if np.issubdtype(generated_test_feats[i].dtype, np.number)]
    print(f"{len(molfeats)} remain after removal invalid types")

    return molfeats


def attach_output_dir_molecule_featurizer(molfeats, conf, create=True, ):
    for f in molfeats:
        f.ddir = os.path.join(conf.DATADIR,
                              inspect.getfile(f.__class__).replace(basedir_molecule_featurizer + os.sep, "").replace(
                                  ".py", ""))
        if create:
            os.makedirs(f.ddir, exist_ok=True)


def get_info(feat) -> Dict:
    target_file = os.path.join(
        feat.ddir,
        f"{feat.__class__.__name__}_feature_info.json"
    )
    feature_info = {}
    if os.path.exists(target_file):
        with open(target_file, "r") as dfile:
            feature_info = json.load(dfile)
    return feature_info


def save_info(feature_info, feat):
    target_file = os.path.join(
        feat.ddir,
        f"{feat.__class__.__name__}_feature_info.json"
    )
    with open(target_file, "w+") as dfile:
        json.dump(feature_info, dfile, indent=4)


def write_info(key, value, feat):
    feature_info = get_info(feat)
    feature_info[key] = value
    save_info(feature_info, feat)
