import os, sys
from typing import List,Tuple
from rdkit.Chem import MolFromSmiles, Mol
from tqdm import tqdm
import pandas as pd
import numpy as np

# import relative if not in sys path
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
from molNet.featurizer._molecule_featurizer import MoleculeFeaturizer, VarSizeMoleculeFeaturizer, \
    prepare_mol_for_featurization
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
    except Exception as e:
        logger.exception(e)
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

def main(dataloader,path,max_mols=None):    
    freeze_support() 
    if dataloader == "ChemBLdb29":
        from molNet.dataloader.molecular.ChEMBLdb import ChemBLdb29 as dataloaderclass
    elif dataloader == "ESOL":
        from molNet.dataloader.molecular.ESOL import ESOL as dataloaderclass
    else:
        raise ValueError(f"unknown dataloader '{dataloader}'")
    mols = None

    for featurizer,mol_to_data in (
            (molNet.featurizer.get_molecule_featurizer_info,lambda x:[x]),
            (molNet.featurizer.get_atom_featurizer_info,lambda x:x.GetAtoms())
    ):
        logger.info("get molNet featurizer")
        featurizer = featurizer()
        logger.info(f"feats  initial length = {len(featurizer)}")

        featurizer["isListFeat"] = featurizer["instance"].apply(lambda f: isinstance(f,FeaturizerList))
        featurizer.drop(featurizer.index[featurizer["isListFeat"]], inplace=True)
        logger.info(f"featurizer length after FeaturizerList drop = {len(featurizer)}")
        featurizer=featurizer[featurizer.dtype!=bool]
        logger.info(f"featurizer length after bool drop = {len(featurizer)}")
        featurizer=featurizer[featurizer.length>0]
        logger.info(f"featurizer length after length<1 drop = {len(featurizer)}")

        featurizer=featurizer.sort_values("length")




        if mols is None:
            loader = dataloaderclass()
            logger.info("laod mols")
            #mols=[None]*loader.expected_data_size
            mols = load_mols(loader,limit=max_mols)

            logger.info("prepare mols")
            mols=[ prepare_mol_for_featurization(m) for m in tqdm(mols)]

        data=[]
        for m in mols:
            data.extend(mol_to_data(m))

        dl_path=os.path.join(path,"raw_features",dataloader)
        featurizer["ecfd_path"]=[os.path.join(dl_path,*mod.split("."))+".dat" for mod in featurizer.index]

        #turns off norm
        featurizer["instance"].apply(lambda i: i.set_preferred_norm("None"))
        featurizer["norem"]=featurizer["instance"].apply(lambda i: i.preferred_norm)


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

        datalength=len(data)
        featurizer["current_length"]= featurizer[["length","current_size"]].apply(
            lambda r: r["current_size"]/datalength,
            axis=1)

        rem_idx=featurizer[~np.isnan(featurizer["current_length"]) & (featurizer["current_length"]!=featurizer["length"])].index
        for p in featurizer.loc[rem_idx]["ecfd_path"]:
             os.remove(p)
        featurizer.loc[rem_idx,"current_length"]=np.nan

        workfeats=featurizer[np.isnan(featurizer["current_length"])].copy()
        if data[0] is None:
            return
        generate_ecdf_dist(data, workfeats)
    
    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-d','--dataloader', type=str,required=True)
    parser.add_argument('--max_mols', type=int)
    parser.add_argument('-p','--path', type=str,default=os.path.join(molNet.get_user_folder(),"autodata","ecdf"))
    args = parser.parse_args()
    main(dataloader=args.dataloader,max_mols=args.max_mols,path=args.path)