import logging
import os, sys
from typing import List,Tuple
from rdkit.Chem import Mol
from tqdm import tqdm
import pandas as pd
import numpy as np
from functools import partial

from multiprocessing import Pool, RLock, freeze_support, current_process, cpu_count

if __name__ == "__main__":
    modp = os.path.dirname(os.path.abspath(__file__))

    while not "molNet" in os.listdir(modp):
        modp=os.path.dirname(modp)
        if os.path.dirname(modp) == modp:
            raise ValueError("connot determine local molNet")
    if modp not in sys.path:
        sys.path.insert(0,modp)
        sys.path.append(modp)

import molNet
from molNet.featurizer.featurizer import FeaturizerList
from molNet.dataloader.molecular.prepmol import PreparedMolDataLoader

logger = molNet.MOLNET_LOGGER
logger.setLevel(logging.DEBUG)

def load_mols(loader, limit=None) -> List[Mol]:
    if limit is not None and limit > 0:
        return loader.get_n_entries(limit)
    return [mol for mol in tqdm(
        loader, unit="mol", unit_scale=True, total=loader.expected_data_size,desc="load mols"
    )]

def limit_featurizer(featurizer,datalength):
    logger.info(f"feats initial length = {len(featurizer)}")

    featurizer["isListFeat"] = featurizer["instance"].apply(lambda f: isinstance(f,FeaturizerList))
    featurizer.drop(featurizer.index[featurizer["isListFeat"]], inplace=True)
    logger.info(f"featurizer length after FeaturizerList drop = {len(featurizer)}")
    featurizer=featurizer[featurizer.dtype!=bool]
    logger.info(f"featurizer length after bool drop = {len(featurizer)}")
    featurizer=featurizer[featurizer.length>0]
    logger.info(f"featurizer length after length<1 drop = {len(featurizer)}")
    featurizer=featurizer.sort_values("length")

    def _cz(r):
        if not os.path.exists(r["ecfd_path"]):
            return np.nan
        try:
            return np.memmap(r["ecfd_path"], dtype=r["dtype"], mode='r',).size
        except (ValueError,FileNotFoundError):
            os.remove(r["ecfd_path"])
            return np.nan



    featurizer["current_size"] = featurizer[["ecfd_path","dtype"]].apply(
        _cz,
        axis=1)

    featurizer["current_length"]= featurizer[["length","current_size"]].apply(
        lambda r: r["current_size"]/datalength,
        axis=1)

    rem_idx=featurizer[~np.isnan(featurizer["current_length"]) & (featurizer["current_length"]!=featurizer["length"])].index
    for p in featurizer.loc[rem_idx]["ecfd_path"]:
        os.remove(p)
    featurizer.loc[rem_idx,"current_length"]=np.nan

    featurizer=featurizer.loc[featurizer.index[np.isnan(featurizer["current_length"])]]

    return featurizer

def generate_ecfd_distr_mol(feat_row,mols,ntotal,pos=None):
    feat=feat_row["instance"]
    path=feat_row["ecfd_path"]
    dtype=feat_row['dtype']
    feat_length=feat_row['length']
    text=f"{feat_row.name.rsplit('.',1)[1]} ({feat_row['idx']}/{ntotal})"

    len_data=len(mols)
    empty_bytes = (np.ones(feat_length)*np.nan).astype(dtype)
    a = np.memmap(path, dtype=dtype, mode='w+', shape=(len_data,feat_length))
    for i,mol in tqdm(enumerate(mols),desc=text,total=len_data, position=pos):
        try:
            r = feat(mol)
            a[i] = r
        except (molNet.ConformerError, ValueError, ZeroDivisionError):
            a[i] = empty_bytes
            pass

def worker(feat_row,mols,ntotal,as_mol=True):
    pos = current_process()._identity[0]-1
    if as_mol:
        return generate_ecfd_distr_mol(feat_row,mols,ntotal=ntotal,pos=pos)


def main(dataloader,path,max_mols=None):
    freeze_support()
    if dataloader == "ChemBLdb29":
        from molNet.dataloader.molecular.ChEMBLdb import ChemBLdb29 as dataloaderclass
    elif dataloader == "ESOL":
        from molNet.dataloader.molecular.ESOL import ESOL as dataloaderclass
    else:
        raise ValueError(f"unknown dataloader '{dataloader}'")

    logger.info("load mols")
    loader = PreparedMolDataLoader(dataloaderclass())
    mols = load_mols(loader,limit=max_mols)

    #for mols
    featurizer=molNet.featurizer.get_molecule_featurizer_info()
    dl_path=os.path.join(path,"raw_features",dataloader)
    featurizer["ecfd_path"]=[os.path.join(dl_path,*mod.split("."))+".dat" for mod in featurizer.index]
    featurizer=limit_featurizer(featurizer,datalength=len(mols))
    featurizer["idx"]=np.arange(featurizer.shape[0])+1

    work=[d for r,d in featurizer.iterrows()]
    with Pool(processes=cpu_count()) as p:
        p.map(partial(worker,mols=mols,ntotal=featurizer.shape[0]), work)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-d','--dataloader', type=str,required=True)
    parser.add_argument('--max_mols', type=int)
    parser.add_argument('-p','--path', type=str,default=os.path.join(molNet.get_user_folder(),"autodata","ecdf"))
    args = parser.parse_args()
    main(dataloader=args.dataloader,max_mols=args.max_mols,path=args.path)




