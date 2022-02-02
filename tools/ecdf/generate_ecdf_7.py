import logging
import os
import random
import sys
import time
from functools import partial
from multiprocessing import Pool, freeze_support, current_process, cpu_count, RLock
from typing import List
import shutil

import numpy as np
from rdkit.Chem import Mol, MolToSmiles, MolToInchiKey, MolFromSmiles
from sqlalchemy.exc import OperationalError
from tqdm import tqdm
import pickle
import json
import sqlalchemy as sql 


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

def ini_limit_featurizer(featurizer):
    logger.info(f"feats initial length = {len(featurizer)}")

    featurizer["isListFeat"] = featurizer["instance"].apply(lambda f: isinstance(f, FeaturizerList))
    featurizer["instance"].apply(lambda f: f.set_preferred_norm(None))
    featurizer.drop(featurizer.index[featurizer["isListFeat"]], inplace=True)
    logger.info(f"featurizer length after FeaturizerList drop = {len(featurizer)}")
    featurizer.drop(featurizer.index[featurizer["dtype"] == bool], inplace=True)
    logger.info(f"featurizer length after bool drop = {len(featurizer)}")
    featurizer.drop(featurizer.index[featurizer["length"] <= 0], inplace=True)
    logger.info(f"featurizer length after length<1 drop = {len(featurizer)}")
    featurizer = featurizer.sort_values("length")

    def _dt(r):
        return np.issubdtype(r["dtype"],np.number)

    featurizer.drop(featurizer.index[~featurizer.apply(_dt,axis=1)], inplace=True)
    logger.info(f"featurizer length after non numeric drop = {len(featurizer)}")

    return featurizer


def update_featurizer(featurizer):
    red_featurizer=featurizer[~featurizer["done"]].copy()
    red_featurizer["done"]=False
    red_featurizer["working"]=False
    for r,d in red_featurizer.iterrows():
        os.makedirs(d["data_path"],exist_ok=True)
        info_file = os.path.join(d["data_path"],"info.json")
        if not os.path.exists(info_file):
            with open(info_file,"w+") as f:
                json.dump({},f,indent=4)

        with open(info_file,"r") as f:
            info = json.load(f)

        info_change=False
        if "done" not in info:
            info["done"]=False
            info_change=True
        else:
            red_featurizer.loc[r,"done"]=info["done"]

        if "working" not in info:
            info["working"]=False
            info_change=True
        else:
            red_featurizer.loc[r,"working"]=info["working"]

        if info_change:
            with open(info_file,"w+") as f:
                json.dump(info,f,indent=4)

    red_featurizer=red_featurizer[~red_featurizer["done"]].copy()

    return red_featurizer

class WorkOnWorkingError(Exception):
    pass

def prep_to_work(row):
    info_file = os.path.join(row["data_path"],"info.json")
    with open(info_file,"r") as f:
        info = json.load(f)
    if info["working"]:
        raise WorkOnWorkingError("try to work on feat wich is under work")
    info["working"]=True
    with open(info_file,"w+") as f:
        json.dump(info,f,indent=4)


def finish_work(row,done):
    info_file = os.path.join(row["data_path"],"info.json")
    with open(info_file,"r") as f:
        info = json.load(f)
    info["working"]=False
    info["done"]=done
    with open(info_file,"w+") as f:
        json.dump(info,f,indent=4)
    lock_file=os.path.join(row["data_path"],".lock")
    if os.path.exists(lock_file):
        os.remove(lock_file)

def get_and_lock_row(featurizer):
    for r,d in featurizer.iterrows():
        if d["working"]:
            continue
        lock_file=os.path.join(d["data_path"],".lock")
        if os.path.exists(lock_file):
            continue
        with open(lock_file,"w+") as f:
            pass
        return d

def featurize_mol(row, mols,n_split=100_000):
    featurizer=row["instance"]
    lenmols=len(mols)

    range_start= np.arange(0,lenmols+n_split,n_split)


    range_start[-1]=lenmols

    files = [os.path.join(row["data_path"],
        f"feats_{range_start[i]+1}_{e}.npy"
                    )
        for i,e in enumerate(range_start[1:])]
    file_sizes=[e-range_start[i]
        for i,e in enumerate(range_start[1:])
    ]

    files_exists=[os.path.exists(f) for f in files]

    if all(files_exists):
        return True

    first_needed_file_index=files_exists.index(False)

    indices=np.zeros(lenmols,dtype=int)
    indices[range_start[:-1]]=1
    indices=np.cumsum(indices)-1

    start_index=(indices==first_needed_file_index).argmax()

    def turnover(current_start_index):
        current_file=files[indices[current_start_index]]
        current_array=(np.zeros((
            file_sizes[indices[current_start_index]],
            row["length"]
        ))*np.nan).astype(row["dtype"])

        ri_max = file_sizes[indices[current_start_index]]

        return current_file, current_array,ri_max,current_start_index,current_start_index+ri_max

    current_file, current_array,ri_max,start_index,stop_index = turnover(start_index)

    ri=0
    for i, mol in tqdm(enumerate(mols), desc="featzurize mols", total=lenmols):
        if i<start_index:
            continue
        current_array[ri]=featurizer(mol)
        ri+=1
        if ri>=ri_max:
            ri=0
            np.save(current_file, current_array)
            if i<lenmols-1:
                while os.path.exists(current_file) and stop_index<lenmols:
                    current_file, current_array,ri_max,start_index,stop_index = turnover(stop_index)

        #print(i,indices[i],files[indices[i]])
    return True





def main(dataloader, path, max_mols=None,ignore_existsing_feats=True,ignore_existsing_data=True,mean_feat_delay=1):
    if dataloader == "ChemBLdb29":
        from molNet.dataloader.molecular.ChEMBLdb import ChemBLdb29 as dataloaderclass
    elif dataloader == "ESOL":
        from molNet.dataloader.molecular.ESOL import ESOL as dataloaderclass
    else:
        raise ValueError(f"unknown dataloader '{dataloader}'")

    path=os.path.abspath(path)
    logger.info(f"using path '{path}'")

    os.makedirs(path,exist_ok=True)

    dataset_name=f"{dataloaderclass.__module__}.{dataloaderclass.__name__}"

    logger.info("load mols")
    loader = PreparedMolDataLoader(dataloaderclass(
        data_streamer_kwargs=dict(iter_None=True))
    )
    mols = MolLoader(loader, limit=max_mols)

    # for mols
    featurizer = molNet.featurizer.get_molecule_featurizer_info()
    logger.info(f"limit featurizer for {dataset_name}")
    featurizer = ini_limit_featurizer(featurizer)
    featurizer["done"]=False

    data_path=os.path.join(path,dataset_name)
    featurizer["data_path"]=[os.path.join(data_path,idx) for idx in featurizer.index]

    red_featurizer=update_featurizer(featurizer)

    while len(red_featurizer)>0:
        row=None
        done=False
        try:
            time.sleep(random.random()*2*mean_feat_delay)
            red_featurizer=update_featurizer(red_featurizer)
            if len(red_featurizer)==0:
                break
            row=get_and_lock_row(red_featurizer)
            if row is None:
                continue
            logger.info(f"featurize {row.name}")
            prep_to_work(row)
            done = featurize_mol(row,mols,)

        except WorkOnWorkingError:
            continue
        except Exception as e:
            print(e)
        finally:
            if row is not None:
                finish_work(row,done)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataloader', type=str, required=True)
    parser.add_argument('--max_mols', type=int)
    parser.add_argument('--path', type=str, default=os.path.join(molNet.get_user_folder(), "autodata", "feats_raw_filebased"))
    args = parser.parse_args()
    main(dataloader=args.dataloader, max_mols=args.max_mols, path=args.path,ignore_existsing_feats=True,ignore_existsing_data=True,)
