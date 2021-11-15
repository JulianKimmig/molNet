import os, sys
from typing import List,Tuple
from rdkit.Chem import MolFromSmiles, Mol
from tqdm import tqdm
import pandas as pd
import numpy as np
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
import molNet.featurizer

from molNet.utils.parallelization.multiprocessing import parallelize
from molNet.featurizer._molecule_featurizer import MoleculeFeaturizer, VarSizeMoleculeFeaturizer
from molNet.featurizer.featurizer import FeaturizerList
from molNet import ConformerError

logger = molNet.MOLNET_LOGGER

test_mol = MolFromSmiles("CCC")

def load_mols(loader, limit=None) -> List[Mol]:
    if limit is not None and limit > 0:
        return loader.get_n_entries(limit)
    return [mol for mol in tqdm(
        loader, unit="mol", unit_scale=True, total=loader.expected_data_size
    )]

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

from multiprocessing import Pool, RLock,freeze_support
from multiprocessing import current_process
from functools import partial

def progresser(f,mols,lenmols,ntotal):
    f,d=f
    text=f"{d.name.rsplit('.',1)[1]} ({d['idx']}/{ntotal})"
    feat=d["instance"]
    path=d["ecfd_path"]
    os.makedirs(os.path.dirname(path),exist_ok=True)
    pos=d['idx']+1
    #pos = current_process()._identity[0]-1
    pos=None
    
    empty_bytes = (np.ones(d["length"])*np.nan).astype(d["dtype"]).tobytes()
    try:
        with open(path,"w+b") as f:
            for i,mol in tqdm(enumerate(mols),desc=text,total=lenmols, position=pos):
                try:
                    f.write(feat(mol).tobytes())
                except (ConformerError, ValueError, ZeroDivisionError):
                    f.write(empty_bytes)
    except:
        os.remove(path)
        return False
    return True
        
def generate_ecdf_dist(mols:List[Mol], molfeats:pd.DataFrame):
    tqdm.set_lock(RLock())
    molfeats["idx"]=np.arange(len(molfeats))+1
    #molfeats=molfeats.iloc[:10]
    call=partial(progresser,mols=mols,lenmols=len(mols),ntotal=len(molfeats))
    for r in molfeats.iterrows():
        if not call(r):
            break
    return
    with Pool(10,initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),)) as p:
        #with tqdm(total=len(molfeats),desc="feas") as pbar:
            for _ in p.map(call, molfeats.iterrows()):
         #       pbar.update(1)
                 pass
    return
    nmols = len(mols)
    for r,d in molfeats.iterrows():
        f=d["instance"]
        logger.info(f"load {f}")
        path=d["ecfd_path"]
        print(path)
        
        os.makedirs(os.path.dirname(path),exist_ok=True)
        fpr = np.memmap(path, dtype=d["dtype"], mode='w+', shape=(nmols,d["length"]))
        
        mol_feats = parallelize(
            _single_call_parallel_featurize_molfiles,
            [(mf, f) for mf in mols],
            cores=1,
            progess_bar=True,
            progress_bar_kwargs=dict(unit=" feats"),
            split_parts=1000,
            target_array=fpr
        )
        
        break
        continue
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
        
def main(dataloader,path,max_mols=None):    
    freeze_support() 
    if dataloader == "ChemBLdb29":
        from molNet.dataloader.molecular.ChEMBLdb import ChemBLdb29 as dataloaderclass
    elif dataloader == "ESOL":
        from molNet.dataloader.molecular.ESOL import ESOL as dataloaderclass
    else:
        raise ValueError(f"unknown dataloader '{dataloader}'")
    
    
    molfeats = molNet.featurizer.get_molecule_featurizer_info()
    molfeats["isListFeat"] = molfeats["instance"].apply(lambda f: isinstance(f,FeaturizerList))
    molfeats.drop(molfeats.index[molfeats["isListFeat"]], inplace=True)
    molfeats=molfeats.sort_values("length")
    loader = dataloaderclass()
    
    logger.info("load mols")
    #mols=[None]*loader.expected_data_size
    mols = load_mols(loader,limit=max_mols)
    
    molfeats=molfeats[molfeats.dtype!=bool]
    molfeats=molfeats[molfeats.length>0]
    
    path=os.path.join(path,"raw_features",dataloader)
    molfeats["ecfd_path"]=[os.path.join(path,*mod.split("."))+".dat" for mod in molfeats.index]

    #turns off norm
    molfeats["instance"].apply(lambda i: i.set_preferred_norm("None"))
    molfeats["norem"]=molfeats["instance"].apply(lambda i: i.preferred_norm)

    #length larfer zero
    molfeats=molfeats[molfeats.length>0]
    #np.memmap(path, dtype=d["dtype"], mode='w+', shape=(lenmols,d["length"]))
    
    def _cz(r):
        if not os.path.exists(r["ecfd_path"]):
            return np.nan
        try:
            return np.memmap(r["ecfd_path"], dtype=r["dtype"], mode='r',).size
        except (ValueError,FileNotFoundError):
            os.remove(r["ecfd_path"])
            return np.nan
            
        
    
    molfeats["current_size"] = molfeats[["ecfd_path","dtype"]].apply(
        _cz,
        axis=1)
    
    l_mols=len(mols)
    molfeats["current_length"]= molfeats[["length","current_size"]].apply(
        lambda r: r["current_size"]/l_mols,
        axis=1)
    #molfeats["zeros"]= molfeats[["ecfd_path","dtype"]].apply(
    #    lambda r : (np.memmap(r["ecfd_path"], dtype=r["dtype"], mode='r',)==0).sum() if os.path.exists(r["ecfd_path"]) else
    #    np.nan,
    #    axis=1
    #)
    rem_idx=molfeats[~np.isnan(molfeats["current_length"]) & (molfeats["current_length"]!=molfeats["length"])].index
    for p in molfeats.loc[rem_idx]["ecfd_path"]:
         os.remove(p)
    molfeats.loc[rem_idx,"current_length"]=np.nan
    
    workfeats=molfeats[np.isnan(molfeats["current_length"])]
    if mols[0] is None:
        return
    generate_ecdf_dist(mols, workfeats)
    
    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-d','--dataloader', type=str,required=True)
    parser.add_argument('--max_mols', type=int)
    parser.add_argument('-p','--path', type=str,default=os.path.join(molNet.get_user_folder(),"autodata","ecdf"))
    args = parser.parse_args()
    main(dataloader=args.dataloader,max_mols=args.max_mols,path=args.path)


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




import sys





import time

IGNORED_FEATURIZER = [  # "GETAWAY_Featurizer",
    # "FpDensityMorgan1_Featurizer",
]



def generate_info(molfeats):
    for f in molfeats:
        print(f"gen info {f}")
        write_info("shape", f(test_mol).shape, f)





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
