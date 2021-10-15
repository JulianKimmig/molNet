import os
import sys

from tools.ecdf._generate_ecdf_helper import _single_call_parallel_featurize_molgraph, test_mol, \
    _single_call_parallel_featurize_molfiles

modp = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, modp)
sys.path.append(modp)

from tools.ecdf import ecdf_conf

import inspect
import pickle

import numpy as np
from typing import List

from rdkit import RDLogger
import sys

from tqdm import tqdm

from molNet.featurizer._molecule_featurizer import MoleculeFeaturizer, VarSizeMoleculeFeaturizer
from molNet.utils.parallelization.multiprocessing import parallelize

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)
import logging

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger('matplotlib').setLevel(logging.WARNING)

from rdkit.Chem import Mol
import json
import time

IGNORED_FEATURIZER = [  # "GETAWAY_Featurizer",
    # "FpDensityMorgan1_Featurizer",
]

basedir_featurizer = os.path.dirname(inspect.getfile(MoleculeFeaturizer))
loader = ecdf_conf.MOL_DATALOADER(ecdf_conf.MOL_DIR)


def get_featurizer():
    import molNet.featurizer.molecule_featurizer as mf

    molfeats = mf._available_featurizer
    print(f"found {len(molfeats)} molecule featurizer")

    # bool is already between 0 and 1
    molfeats = [f for f in molfeats if f.dtype != bool]
    print(f"{len(molfeats)} remain after removal of bool types")
    molfeats = [f for f in molfeats if str(f) not in IGNORED_FEATURIZER]
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


def attach_output_dir(molfeats):
    for f in molfeats:
        f.ddir = os.path.join(ecdf_conf.DATADIR,
                              inspect.getfile(f.__class__).replace(basedir_featurizer + os.sep, "").replace(".py", ""))
        os.makedirs(f.ddir, exist_ok=True)


def generate_info(molfeats):
    for f in molfeats:
        print(f"gen info {f}")
        write_info("shape", f(test_mol).shape, f)


def load_mols() -> List[Mol]:
    if ecdf_conf.LIMIT_MOLECULES is not None:
        return loader.get_n_entries(ecdf_conf.LIMIT_MOLECULES)
    return [mol for mol in tqdm(
        loader, unit="mol", unit_scale=True, total=loader.expected_data_size
    )]


def write_info(key, value, feat):
    target_file = os.path.join(
        feat.ddir,
        f"{feat.__class__.__name__}_feature_info.json"
    )
    feature_info = {}
    if os.path.exists(target_file):
        with open(target_file, "r") as dfile:
            feature_info = json.load(dfile)

    feature_info[key] = value
    with open(target_file, "w+") as dfile:
        json.dump(feature_info, dfile, indent=4)


def generate_ecdf_dist(mols, molfeats):
    nmols = len(mols)
    for f in molfeats:
        print(f"load {f}")

        target_file = os.path.join(
            f.ddir,
            f"{f.__class__.__name__}_feature_dist.pckl"
        )

        if os.path.exists(target_file):
            continue

        ts = time.time()
        mol_feats = parallelize(
            _single_call_parallel_featurize_molfiles,
            [(mf, f) for mf in mols],
            cores="all-1",
            progess_bar=True,
            progress_bar_kwargs=dict(unit=" feats"),
            split_parts=1000
        )
        te = time.time()

        with open(target_file, "w+b") as dfile:
            pickle.dump(mol_feats, dfile)

        write_info(key="time", value=(te - ts) / nmols, feat=f)


def main():
    molfeats = get_featurizer()
    attach_output_dir(molfeats)
    generate_info(molfeats)

    mols = load_mols()

    generate_ecdf_dist(mols, molfeats)


if __name__ == '__main__':
    main()
