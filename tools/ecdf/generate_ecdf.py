import os, sys
if __name__ == "__main__":
    modp = os.path.dirname(os.path.dirname(os.path.abspath("")))
    while not "molNet" in os.listdir(modp):
        modp=os.path.dirnmae(modp)
        
    if modp not in sys.path:
        sys.path.insert(0,modp)
        sys.path.append(modp)

import molNet
import molNet.featurizer

def main():
    molNet.featurizer.get_molecule_featurizer_info()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-p','--path', type=str)
    args = parser.parse_args()
    main()


quit()

import gzip
import os
import sys

modp = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(modp)
sys.path.insert(0, modp)
sys.path.append(modp)

from tools.ecdf._generate_ecdf_helper import test_mol, \
    _single_call_parallel_featurize_molfiles, get_molecule_featurizer, attach_output_dir_molecule_featurizer, \
    write_info, _single_call_check_distributionfiles

import pickle

from typing import List

from rdkit import RDLogger
import sys

from tqdm import tqdm

from molNet.utils.parallelization.multiprocessing import parallelize

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)
import logging

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger('matplotlib').setLevel(logging.WARNING)

from rdkit.Chem import Mol
import time

IGNORED_FEATURIZER = [  # "GETAWAY_Featurizer",
    # "FpDensityMorgan1_Featurizer",
]



def generate_info(molfeats):
    for f in molfeats:
        print(f"gen info {f}")
        write_info("shape", f(test_mol).shape, f)


def load_mols(loader, conf) -> List[Mol]:
    if conf.LIMIT_MOLECULES is not None and conf.LIMIT_MOLECULES > 0:
        return loader.get_n_entries(conf.LIMIT_MOLECULES)
    return [mol for mol in tqdm(
        loader, unit="mol", unit_scale=True, total=loader.expected_data_size
    )]


def check_preexisting(molfeats):
    to_work = parallelize(
        _single_call_check_distributionfiles,
        molfeats,
        cores="all-1",
        progess_bar=True,
        progress_bar_kwargs=dict(unit=" feats"),
        split_parts=1000
    )
    return to_work

def generate_ecdf_dist(mols, molfeats):
    nmols = len(mols)
    for f in molfeats:
        print(f"load {f}")

        if os.path.exists(f.feature_dist_gpckl):
            continue

        if os.path.exists(f.feature_dist_pckl):
            with open(f.feature_dist_pckl, "rb") as dfile:
                mol_feats = pickle.load(dfile)

            with gzip.open(f.feature_dist_gpckl, "w+b") as dfile:
                pickle.dump(mol_feats, dfile)
            os.remove(f.feature_dist_pckl)
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

        with gzip.open(f.feature_dist_gpckl, "w+b") as dfile:
            pickle.dump(mol_feats, dfile)

        write_info(key="time", value=(te - ts) / nmols, feat=f)


def main():
    from tools.ecdf import ecdf_conf

    loader = ecdf_conf.MOL_DATALOADER(ecdf_conf.MOL_DIR)
    molfeats = get_molecule_featurizer(ignored_names=IGNORED_FEATURIZER)
    attach_output_dir_molecule_featurizer(molfeats, ecdf_conf)

    generate_info(molfeats)

    molfeats = check_preexisting(molfeats)
    print(molfeats)
    if len(molfeats) == 0:
        print("no more feats")
        return
    mols = load_mols(loader, ecdf_conf)

    generate_ecdf_dist(mols, molfeats)


if __name__ == '__main__':
    main()
