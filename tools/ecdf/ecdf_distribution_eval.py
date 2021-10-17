import argparse
import gzip
import os
import sys


modp = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, modp)
sys.path.append(modp)

from molNet.utils.parallelization.multiprocessing import parallelize


from tools.ecdf._generate_ecdf_helper import get_molecule_featurizer, attach_output_dir_molecule_featurizer, \
    _single_call_gen_ecdf_images

import pickle


def main(cores):
    from tools.ecdf import ecdf_conf
    mf = get_molecule_featurizer(check_number=False)
    attach_output_dir_molecule_featurizer(mf, ecdf_conf, create=False)

    for f in mf:
        if os.path.exists(f.feature_dist_pckl):
            if not os.path.exists(f.feature_dist_gpckl):
                with open(f.feature_dist_pckl, "rb") as dfile:
                    mol_feats = pickle.load(dfile)

                with gzip.open(f.feature_dist_gpckl, "w+b") as dfile:
                    pickle.dump(mol_feats, dfile)
            os.remove(f.feature_dist_pckl)

    mf = [f for f in mf if os.path.exists(f.feature_dist_gpckl)]

    to_work = parallelize(
        _single_call_gen_ecdf_images,
        mf,
        cores=cores,
        progess_bar=True,
        progress_bar_kwargs=dict(unit=" feats"),
        split_parts=1000
    )
    return to_work


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-c','--cores', type=str, help='cores for worker',default="all-1")

    args = parser.parse_args()
    main(**vars(args))