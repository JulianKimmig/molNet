import gzip
import inspect
import json
import os
import pickle
from typing import List, Tuple, Dict

import numpy as np
from matplotlib import pyplot as plt
from rdkit.Chem import MolFromSmiles, Mol

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

def _single_call_check_distributionfiles(d):
    to_work = []
    for f in d:
        if os.path.exists(f.feature_dist_gpckl):
            continue

        if os.path.exists(f.feature_dist_pckl):
            with open(f.feature_dist_pckl, "rb") as dfile:
                mol_feats = pickle.load(dfile)

            with gzip.open(f.feature_dist_gpckl, "w+b") as dfile:
                pickle.dump(mol_feats, dfile)
            os.remove(f.feature_dist_pckl)
            continue

        to_work.append(f)
    return to_work

def _single_call_parallel_featurize_molfiles(d: Tuple[Mol, MoleculeFeaturizer]):
    feat = d[0][1]
    r = np.zeros((len(d), *feat(test_mol).shape)) * np.nan
    for i, data in enumerate(d):
        mol = data[0]
        feat = data[1]
        feat.preferred_norm = None
        try:
            r[i] = feat(mol)
        except (ConformerError, ValueError, ZeroDivisionError):
            pass
    return r


def get_molecule_featurizer(ignored_names=None, ignore_var_size=True, check_number=True, check_bool=True):
    if ignored_names is None:
        ignored_names = []
    import molNet.featurizer.molecule_featurizer as mf

    molfeats = mf._available_featurizer
    print(f"found {len(molfeats)} molecule featurizer")

    if check_bool:
        # bool is already between 0 and 1
        molfeats = [f for f in molfeats if f.dtype != bool]
    if len(ignored_names) > 0:
        print(f"{len(molfeats)} remain after removal of bool types")
        molfeats = [f for f in molfeats if str(f) not in ignored_names]

    if ignore_var_size:
        molfeats = [f for f in molfeats if not isinstance(f, VarSizeMoleculeFeaturizer)]
        print(f"{len(molfeats)} remain after removal of ignored")

    if check_number:
        generated_test_feats = parallelize(
            _single_call_parallel_featurize_molgraph,
            molfeats,
            cores="all-1",
            progess_bar=True,
            progress_bar_kwargs=dict(unit=" feats"),
        )

        molfeats = [molfeats[i] for i in range(len(molfeats)) if
                    np.issubdtype(generated_test_feats[i].dtype, np.number)]
        print(f"{len(molfeats)} remain after removal invalid types")

    return molfeats


def attach_output_dir_molecule_featurizer(molfeats, conf, create=True, ):
    for f in molfeats:
        f.ddir = os.path.join(os.path.abspath(conf.DATADIR),
                              inspect.getfile(f.__class__).replace(basedir_molecule_featurizer + os.sep, "").replace(
                                  ".py", ""))
        if create:
            os.makedirs(f.ddir, exist_ok=True)

        f.feature_dist_pckl = os.path.join(
            f.ddir,
            f"{f.__class__.__name__}_feature_dist.pckl"
        )

        f.feature_dist_gpckl = os.path.join(
            f.ddir,
            f"{f.__class__.__name__}_feature_dist.pckl.gz"
        )

        f.feature_dist_npz = os.path.join(
            f.ddir,
            f"{f.__class__.__name__}_feature_dist.npz"
        )


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


def generate_ecdf(data, res_1_99=None, smooth=False, unique_only=False):
    if data.ndim > 1:
        data = np.squeeze(data)
        if data.ndim > 1:
            return [
                generate_ecdf(data[..., i], res_1_99=res_1_99, smooth=smooth, unique_only=unique_only)
                for i in range(data.shape[-1])
            ]
    x = np.sort(data)
    n = len(data)
    y = np.arange(1, n + 1) / n
    if smooth:
        unique_only = True
        x, uindices = np.unique(x, return_index=True)
        y = np.array([a.mean() for a in np.split(y, uindices[1:])])
        y[0] = 0
        y[-1] = 1

    if res_1_99:
        ix1=(y >= 0.01).argmin()
        ix99=(y >= 0.99).argmin()
        dix1=0
        dix99=0
        x1 = x[ix1]
        x99 = x[ix99]
        while ix1>=0 and ix99<len(x) and x1==x99:
            if dix1<=dix99:
                dix1+=1
                ix1-=1
                if ix1<=0:
                    dix1=np.inf
                    ix1=0
            else:
                dix99+=9
                ix99+=1
            x1 = x[ix1]
            x99 = x[ix99]
        print(ix1,ix99,len(x))
        if x1 != x99:
            res = res_1_99 / (x99 - x1)  # ppu
            print(res_1_99,x99,x1,x[0],x[-1])
            points = int((x[-1] - x[0]) * res)
            dp = np.round((np.linspace(0, (len(x) - 1), points))).astype(int)
            x = x[dp]
            y = y[dp]
        raise ValueError()

    if unique_only:
        x, uindices = np.unique(x, return_index=True)
        y = y[uindices]
    return x, y


def _single_call_gen_ecdf_images(mf):
    paths = []
    for f in mf:
        print(f)
        eg = ECDFGroup(f.feature_dist_gpckl, save_full_data=False, save_smooth_data=True)

        # print(eg.dist_data.shape)
        paths.extend(eg.get_ecdf_img_paths())
    return paths


class ECDFGroup():
    def __init__(self, feat_dist_file, save_full_data=True, save_smooth_data=True):
        self.dirname = os.path.dirname(os.path.abspath(feat_dist_file))
        self.dist_pickle = os.path.basename(feat_dist_file)
        self.basename = self.dist_pickle.replace("_feature_dist.pckl", "").replace(".gz", "")

        self.save_smooth_data = save_smooth_data
        self.save_full_data = save_full_data
        self.feat_dist_file = feat_dist_file
        self.ecdfs: List["ECDF"] = []
        self.need_save = False

    def check_ecdfs(self):
        if len(self.ecdfs) == 0:
            self.full_data = None
            self.smooth_data = None
            with gzip.open(self.feat_dist_file, "rb") as f:
                feat_dist = pickle.load(f)

            if len(feat_dist.shape) > 2:
                raise NotImplementedError(
                    "cannot work with larger than 1d features, do they make sense? can you split them?")
            self.feat_dist = feat_dist[~np.isnan(feat_dist).any(1)]
            self.ecdfs = []

            def gen_smooth_getter(k):
                def getter():
                    return self._get_sub_smooth_data(k)

                return getter

            def gen_full_getter(k):
                def getter():
                    return self._get_sub_full_data(k)

                return getter

            for k in range(feat_dist.shape[1]):
                ecdf = ECDF(
                    feat_dist_file=self.feat_dist_file,
                    n=k,
                    dist_data=feat_dist[:, k],
                    save_full_data=False,
                    save_smooth_data=False,
                )
                ecdf.get_smooth_ecdf = gen_smooth_getter(k)
                ecdf.get_full_ecdf = gen_full_getter(k)
                self.ecdfs.append(ecdf)

        pass

    def __str__(self):
        return f"{self.basename}"

    @property
    def dist_data(self):
        self.check_ecdfs()
        d = np.array([ecdf.dist_data for ecdf in self.ecdfs])
        self.save()
        return d

    def get_ecdf_img_paths(self):
        self.check_ecdfs()
        paths = [ecdf.get_ecdf_img_path() for ecdf in self.ecdfs]
        self.save()
        return paths

    def save(self):
        if not self.need_save:
            return
        if self.full_data is not None:
            if self.save_full_data:
                path = os.path.join(self.dirname, str(self) + "_full_ecdf.pckl.gz")
                with gzip.open(path, "w+b") as f:
                    pickle.dump(self.full_data, f)

        if self.smooth_data is not None:
            if self.save_smooth_data:
                path = os.path.join(self.dirname, str(self) + "_smooth_ecdf.pckl.gz")
                with gzip.open(path, "w+b") as f:
                    pickle.dump(self.smooth_data, f)
        self.need_save = False

    def _get_sub_full_data(self, k):
        self.check_ecdfs()
        path = os.path.join(self.dirname, str(self) + "_full_ecdf.pckl.gz")
        if self.full_data is None:
            self.full_data = {i: None for i in range(len(self.ecdfs))}
            if os.path.exists(path):
                with gzip.open(path, "rb") as f:
                    self.full_data.update(pickle.load(f))

        if self.full_data[k] is None:
            self.need_save = True
            print(f"gen full ecdf for {self}[{k}]")
            x, y = generate_ecdf(self.feat_dist[:, k], smooth=False, unique_only=False)
            self.full_data[k] = (x, y)

        if self.full_data[k] is None:
            raise ValueError()

        return self.full_data[k]

    def get_smooth_data(self):
        self.check_ecdfs()
        paths = [ecdf.smooth_ecdf for ecdf in self.ecdfs]
        self.save()
        return paths

    def _get_sub_smooth_data(self, k):
        self.check_ecdfs()
        path = os.path.join(self.dirname, str(self) + "_smooth_ecdf.pckl.gz")
        if self.smooth_data is None:
            self.smooth_data = {i: None for i in range(len(self.ecdfs))}

            if os.path.exists(path):
                with gzip.open(path, "rb") as f:
                    self.smooth_data.update(pickle.load(f))

        if self.smooth_data[k] is None:
            self.need_save = True
            print(f"gen smooth ecdf for {self}[{k}]")
            x, y = generate_ecdf(self.feat_dist[:, k], res_1_99=10_000, smooth=True)
            self.smooth_data[k] = (x, y)

        if self.smooth_data[k] is None:
            raise ValueError()

        return self.smooth_data[k]


class ECDF:
    def __init__(self, feat_dist_file, n=0, dist_data=None, save_full_data=True, save_smooth_data=True):
        self.dirname = os.path.dirname(os.path.abspath(feat_dist_file))
        self.dist_pickle = os.path.basename(feat_dist_file)
        self.basename = self.dist_pickle.replace("_feature_dist.pckl", "").replace(".gz", "")
        self.save_full_data = save_full_data
        self.save_smooth_data = save_smooth_data
        self._n = n
        self._dist_data = dist_data

    def __str__(self):
        return f"{self.basename}_{self._n}"

    @property
    def dist_data(self):
        if self._dist_data is not None:
            return self._dist_data

        print(self, "load dist_data")
        with gzip.open(os.path.join(self.dirname, self.dist_pickle), "rb") as f:
            feat_dist = pickle.load(f)

        if len(feat_dist.shape) > 2:
            raise NotImplementedError(
                "cannot work with larger than 1d features, do they make sense? can you split them?")

        feat_dist = feat_dist[~np.isnan(feat_dist).any(1)]

        self._dist_data = feat_dist[:, self._n]

        return self._dist_data

    def get_full_ecdf(self):
        path = os.path.join(self.dirname, str(self) + "_full_ecdf.pckl.gz")
        x, y = None, None
        if os.path.exists(path):
            with gzip.open(path, "rb") as f:
                x, y = pickle.load(f)
            if self.dist_data.shape[0] != x.shape[0] or self.dist_data.shape[0] != y.shape[0]:
                x, y = None, None

        if x is None or y is None:
            print(self, "generate full_ecdf")
            x, y = generate_ecdf(self.dist_data, smooth=False, unique_only=False)
            if self.save_full_data:
                with gzip.open(path, "w+b") as f:
                    pickle.dump((x, y), f)
        return x, y

    @property
    def full_ecdf(self):
        return self.get_full_ecdf()

    def get_smooth_ecdf(self):
        path = os.path.join(self.dirname, str(self) + "_smooth_ecdf.pckl.gz")
        x, y = None, None
        if os.path.exists(path):
            with gzip.open(path, "rb") as f:
                x, y = pickle.load(f)
            if self.dist_data.max() != x.max() or self.dist_data.min() != x.min() or y.max() != 1 or y.min() != 0:
                x, y = None, None

        if x is None or y is None:
            print(self, "generate smooth_ecdf")
            x, y = generate_ecdf(self.dist_data, res_1_99=10_000, smooth=True)
            if self.save_smooth_data:
                with gzip.open(path, "w+b") as f:
                    pickle.dump((x, y), f)

        return x, y

    @property
    def smooth_ecdf(self):
        return self.get_smooth_ecdf()

    def get_ecdf_img_path(self,create_if_not_exist=True):
        path = os.path.join(self.dirname, str(self) + "_ecdf.png")
        if not os.path.exists(path) and create_if_not_exist:
            print(f"generate image for {self}")
            plt.plot(*self.full_ecdf, label="ECDF")
            plt.plot(*self.smooth_ecdf, label="smoothed ECDF")
            plt.legend()
            plt.savefig(path)
            plt.close()
        return path
