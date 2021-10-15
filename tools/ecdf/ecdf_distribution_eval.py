import gzip
import os
import sys

modp = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, modp)
sys.path.append(modp)

from tools.ecdf._generate_ecdf_helper import get_molecule_featurizer, attach_output_dir_molecule_featurizer, \
    ECDFGroup

import pickle


def main():
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

    for f in mf:
        print(f)
        eg = ECDFGroup(f.feature_dist_gpckl, save_full_data=False, save_smooth_data=True)

        print(eg.dist_data.shape)
        print(eg.get_smooth_data())


if __name__ == '__main__':
    main()
