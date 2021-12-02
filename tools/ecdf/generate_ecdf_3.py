import logging
import os
import sys
from functools import partial
from multiprocessing import Pool, freeze_support, current_process, cpu_count, RLock
from typing import List

import numpy as np
from rdkit.Chem import Mol
from tqdm import tqdm

if __name__ == "__main__":
    modp = os.path.dirname(os.path.abspath(__file__))

    while not "molNet" in os.listdir(modp):
        modp = os.path.dirname(modp)
        if os.path.dirname(modp) == modp:
            raise ValueError("connot determine local molNet")
    if modp not in sys.path:
        sys.path.insert(0, modp)
        sys.path.append(modp)

    import molNet
    from molNet.featurizer.featurizer import FeaturizerList
    from molNet.dataloader.molecular.prepmol import PreparedMolDataLoader

    logger = molNet.MOLNET_LOGGER
    logger.setLevel(logging.DEBUG)





def limit_featurizer(featurizer, datalength):
    logger.info(f"feats initial length = {len(featurizer)}")

    featurizer["isListFeat"] = featurizer["instance"].apply(lambda f: isinstance(f, FeaturizerList))
    featurizer.drop(featurizer.index[featurizer["isListFeat"]], inplace=True)
    logger.info(f"featurizer length after FeaturizerList drop = {len(featurizer)}")
    featurizer = featurizer[featurizer.dtype != bool]
    logger.info(f"featurizer length after bool drop = {len(featurizer)}")
    featurizer = featurizer[featurizer.length > 0]
    logger.info(f"featurizer length after length<1 drop = {len(featurizer)}")
    featurizer = featurizer.sort_values("length")

    def _cz(r):
        if not os.path.exists(r["ecfd_path"]):
            return np.nan
        if os.path.exists(r["ecfd_path"]+"_block"):
            print(r)
            return r["length"]*datalength
        try:
            return np.memmap(r["ecfd_path"], dtype=r["dtype"], mode='r', ).size
        except (ValueError, FileNotFoundError):
            os.remove(r["ecfd_path"])
            return np.nan

    featurizer["current_size"] = featurizer[["ecfd_path", "dtype","length"]].apply(
        _cz,
        axis=1)

    featurizer["current_length"] = featurizer[["length", "current_size"]].apply(
        lambda r: r["current_size"] / datalength,
        axis=1)

    rem_idx = featurizer[
        ~np.isnan(featurizer["current_length"]) & (featurizer["current_length"] != featurizer["length"])].index
    for p in featurizer.loc[rem_idx]["ecfd_path"]:
        os.remove(p)
    featurizer.loc[rem_idx, "current_length"] = np.nan

    featurizer = featurizer.loc[featurizer.index[np.isnan(featurizer["current_length"])]]

    logger.info(f"featurizer length after invalid data drop = {len(featurizer)}")

    return featurizer

def pre_generate_ecfd_distr(feat_row,len_data,ntotal):
    feat = feat_row["instance"]
    path = feat_row["ecfd_path"]
    dtype = feat_row['dtype']
    feat_length = feat_row['length']
    text = f"{feat_row.name.rsplit('.', 1)[1]} ({feat_row['idx']}/{ntotal})"

    empty_bytes = (np.ones(feat_length) * np.nan).astype(dtype)
    os.makedirs(os.path.dirname(path),exist_ok=True)
    with open(path+"_block","w+") as f:
        pass

    a = np.memmap(path, dtype=dtype, mode='w+', shape=(len_data, feat_length))
    return feat,text,empty_bytes,a,path

def post_generate_ecfd_distr(path):
    os.remove(path+"_block")

def generate_ecfd_distr_mol(feat_row, mols, ntotal, pos=None,len_data=None):
    if len_data is None:
        len_data = len(mols)
    feat,text,empty_bytes,a,path = pre_generate_ecfd_distr(feat_row,len_data,ntotal)
    for i, mol in tqdm(enumerate(mols), desc=text, total=len_data, position=pos):
        try:
            r = feat(mol)
            a[i] = r
        except (molNet.ConformerError, ValueError, ZeroDivisionError):
            a[i] = empty_bytes
            pass
    post_generate_ecfd_distr(path)

def generate_ecfd_distr_atom(feat_row, mols, ntotal, pos=None,len_data=None):
    if len_data is None:
        len_data=0
        for m in mols:
            len_data+=m.GetNumAtoms()

    feat,text,empty_bytes,a,path = pre_generate_ecfd_distr(feat_row,len_data,ntotal)

    for i, mol in tqdm(enumerate(mols), desc=text, total=len(mols), position=pos):
        for at in mol.GetAtoms():
            try:
                r = feat(at)
                a[i] = r
            except (molNet.ConformerError, ValueError, ZeroDivisionError):
                a[i] = empty_bytes
                pass
    post_generate_ecfd_distr(path)



def worker(feat_row, mols, ntotal, as_atom=False):
    pos = current_process()._identity[0] - 1
    if as_atom:
        return generate_ecfd_distr_atom(feat_row, mols, ntotal=ntotal, pos=pos)
    else:
        return generate_ecfd_distr_mol(feat_row, mols, ntotal=ntotal, pos=pos)

def load_mols(loader, limit=None) -> List[Mol]:
    if limit is not None and limit > 0:
        return loader.get_n_entries(limit)
    return [mol for mol in tqdm(
        loader, unit="mol", unit_scale=True, total=loader.expected_data_size, desc="load mols"
    )]

class MolLoader():
    def __init__(self,molloader, limit=None):
        self.limit = limit
        self.molloader = molloader

    def __len__(self):
        limit=self.limit if (self.limit is not None and self.limit>0) else self.molloader.expected_data_size
        if limit<self.molloader.expected_data_size:
            return limit
        return self.molloader.expected_data_size

    def __iter__(self):
        if self.limit is not None and self.limit>0:
            return (m for m in self.molloader.get_n_entries(self.limit))
        return iter(tqdm(
            self.molloader, unit="mol", unit_scale=True, total=self.molloader.expected_data_size, desc="load mols"
        ))

def main(dataloader, path, max_mols=None):
    freeze_support()
    if dataloader == "ChemBLdb29":
        from molNet.dataloader.molecular.ChEMBLdb import ChemBLdb29 as dataloaderclass
    elif dataloader == "ESOL":
        from molNet.dataloader.molecular.ESOL import ESOL as dataloaderclass
    else:
        raise ValueError(f"unknown dataloader '{dataloader}'")

    logger.info("load mols")
    loader = PreparedMolDataLoader(dataloaderclass())
    mols = MolLoader(loader, limit=max_mols)

    # for mols
    datalength=len(mols)
    featurizer = molNet.featurizer.get_molecule_featurizer_info()
    dl_path = os.path.join(path, "raw_features", dataloader)
    featurizer["ecfd_path"] = [os.path.join(dl_path, *mod.split(".")) + ".dat" for mod in featurizer.index]
    featurizer = limit_featurizer(featurizer, datalength=datalength)
    featurizer["idx"] = np.arange(featurizer.shape[0]) + 1

    while len(featurizer)>0:
        logger.info(f"featurize {featurizer.iloc[0].name}")
        generate_ecfd_distr_mol(featurizer.iloc[0], mols, ntotal=featurizer.shape[0], pos=None,len_data=datalength)
        featurizer = limit_featurizer(featurizer, datalength=datalength)
        featurizer["idx"] = np.arange(featurizer.shape[0]) + 1


    # for atoms
    datalength=sum([m.GetNumAtoms() for m in mols])
    featurizer = molNet.featurizer.get_atom_featurizer_info()
    dl_path = os.path.join(path, "raw_features", dataloader)
    featurizer["ecfd_path"] = [os.path.join(dl_path, *mod.split(".")) + ".dat" for mod in featurizer.index]
    featurizer = limit_featurizer(featurizer, datalength=datalength)
    featurizer["idx"] = np.arange(featurizer.shape[0]) + 1

    while len(featurizer)>0:
        logger.info(f"featurize {featurizer.iloc[0].name}")
        generate_ecfd_distr_atom(featurizer.iloc[0], mols, ntotal=featurizer.shape[0], pos=None,len_data=datalength)
        featurizer = limit_featurizer(featurizer, datalength=datalength)
        featurizer["idx"] = np.arange(featurizer.shape[0]) + 1

    logger.info("nothing to do")



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataloader', type=str, required=True)
    parser.add_argument('--max_mols', type=int)
    parser.add_argument('-p', '--path', type=str, default=os.path.join(molNet.get_user_folder(), "autodata", "ecdf"))
    args = parser.parse_args()
    main(dataloader=args.dataloader, max_mols=args.max_mols, path=args.path)
