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
from rdkit.Chem import Mol, MolToSmiles, MolToInchiKey
from tqdm import tqdm
import pickle
import json

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

def limit_featurizer(featurizer, datalength,dl_name,ignore_existsing_feats=True,ignore_working_ds=True):
    
    logger.info(f"limit featurizer for {dl_name}")
    logger.info(f"feats initial length = {len(featurizer)}")

    featurizer["isListFeat"] = featurizer["instance"].apply(lambda f: isinstance(f, FeaturizerList))
    featurizer.drop(featurizer.index[featurizer["isListFeat"]], inplace=True)
    logger.info(f"featurizer length after FeaturizerList drop = {len(featurizer)}")
    featurizer = featurizer[featurizer.dtype != bool]
    logger.info(f"featurizer length after bool drop = {len(featurizer)}")
    featurizer = featurizer[featurizer.length > 0]
    logger.info(f"featurizer length after length<1 drop = {len(featurizer)}")
    featurizer = featurizer.sort_values("length")
    
    if ignore_working_ds:
        featurizer["isworking"]=False
        for r,d in featurizer.iterrows():
            wf = os.path.join(d["path"],f"{dl_name}.work")
            if os.path.exists(wf):
                featurizer.loc[r,"isworking"]=True
        
        featurizer = featurizer[~featurizer["isworking"]]
        logger.info(f"featurizer length after working = {len(featurizer)}")
        
    if ignore_existsing_feats:
        featurizer["isin"]=False
        for r,d in featurizer.iterrows():
            dsjson = os.path.join(d["path"],"datasets.json")
            if os.path.exists(dsjson):
                with open(dsjson,"r") as f:
                    ds=json.loads(f.read())
                if dl_name in ds:
                    featurizer.loc[r,"isin"]=True
        
        featurizer = featurizer[~featurizer["isin"]]
        logger.info(f"featurizer length after existing = {len(featurizer)}")
    
    return featurizer

def pre_generate_ecfd_distr(feat_row,dl_name):
    feat = feat_row["instance"]
    path = feat_row["path"]
    text = f"{feat_row.name.rsplit('.', 1)[1]}"
    path_data=os.path.join(path,"data")
    #os.makedirs(path,exist_ok=True)
    os.makedirs(path_data,exist_ok=True)
    wf = os.path.join(path,f"{dl_name}.work")
    with open(wf,"w+") as f:
        pass
    #print(path_data)
    db_lookup=os.path.join(path,"lookup.pckl")
    if not os.path.exists(db_lookup):
        db_lookup = {}
    else:
        with open(db_lookup,"rb") as f:
            db_lookup = pickle.load(f)
            
    return feat,path,path_data,db_lookup,text

def post_generate_data(path,db_lookup,change,dl_name,files_in):
    wf = os.path.join(path,f"{dl_name}.work")
    if os.path.exists(wf):
        os.remove(wf)
    if change:
        db_lookup_path=os.path.join(path,"lookup.pckl")
        with open(db_lookup_path,"w+b") as f:
                pickle.dump(db_lookup,f)
    
    dsjson = os.path.join(path,"datasets.json")
    if not os.path.exists(dsjson):
        ds = json.dumps({},indent=4)
        with open(dsjson,"w+") as f:
            f.write(ds)
    with open(dsjson,"r") as f:
        d=f.read()
    d=json.loads(d)
    
    if not dl_name in d or d[dl_name]!=files_in:
        
        
        with open(dsjson,"r") as f:
            d=f.read()
        d=json.loads(d)
        d[dl_name]=files_in
        ds = json.dumps(d,indent=4)
        with open(dsjson,"w+") as f:
            f.write(ds)
            

def hash_array(a):
    return "".join(str(x) for x in list(a.shape)+[int(a.sum()),int(a.mean())])

def raw_feat_mols(feat_row,mols,dl_name,len_data=None, pos=None,ignore_existsing_data=False,inchies=None):
    if len_data is None:
        len_data = len(mols)
    feat,path,path_data,db_lookup,text = pre_generate_ecfd_distr(feat_row,dl_name)
    db_lookup_files=list(db_lookup.values())
    #print(db_lookup_files)
    #print(db_lookup)
    if inchies is None:
        inchies=[]
    skip=[]
    if ignore_existsing_data:
        for i,inchikey in enumerate(inchies):
            if inchikey in db_lookup:
                skip.append(i)
    files_in=len(skip)
    print(files_in,len_data)
    n=0
    change=False
    try:
        for i, mol in tqdm(enumerate(mols), desc=text, total=len_data, position=pos):
            if i in skip:
                continue
            inchikey=MolToInchiKey(mol)
            if inchikey in db_lookup:
                if ignore_existsing_data:
                    files_in+=1
                    continue
                else:
                    n=db_lookup[inchikey]
                    db_lookup_files.remove(n)
            
            try:
                while n in db_lookup_files:
                    n+=1
                r = feat(mol)
                ha=hash_array(r)
                
                
                np.save(os.path.join(path_data,str(n)),r)
                db_lookup_files.append(n)
                db_lookup[inchikey]=n
                change=True
                files_in+=1
                n+=1
            except (molNet.ConformerError, ValueError, ZeroDivisionError) as e:
                print(e)
                
    except Exception as e:
        print(e)
       # shutil.rmtree(path)
    finally:
        post_generate_data(path,db_lookup,change,dl_name,files_in)
    print(n,files_in)
    return True
    #raise ValueError("AA")
    
def main(dataloader, path, max_mols=None,ignore_existsing_feats=True,ignore_existsing_data=True,):
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
    #dl_path = os.path.join(path, "raw_features", dataloader)
    dl_name=f"{dataloaderclass.__module__}.{dataloaderclass.__name__}"
    
    print(featurizer.index[0])
    featurizer["path"] = [os.path.join(path, *mod.split(".")) for mod in featurizer.index]
    
    
   
    inchifile=os.path.join(molNet.get_user_folder(), "autodata", "dataloader",dl_name,"inchies.json")
    inchies = []
    if os.path.exists(inchifile):
        with open(inchifile,"r") as f:
            inchies = json.load(f)
    if not abs(len(inchies)-len(mols))<0.1*len(mols):
        inchies = []
        for i, mol in tqdm(enumerate(mols), desc="load inchies", total=datalength):
                inchies.append(MolToInchiKey(mol))
        os.makedirs(os.path.dirname(inchifile),exist_ok=True)
        with open(inchifile,"w+") as f:
            json.dump(inchies,f)
            
    featurizer = limit_featurizer(featurizer, datalength=datalength,dl_name=dl_name,ignore_existsing_feats=ignore_existsing_feats)
    featurizer["idx"] = np.arange(featurizer.shape[0]) + 1
    while len(featurizer)>0:
        logger.info(f"featurize {featurizer.iloc[0].name}")
        r = raw_feat_mols(featurizer.iloc[0],mols,dl_name,ignore_existsing_data=ignore_existsing_data,inchies=inchies)
        if r:
            featurizer.drop(featurizer.index[0],inplace=True)
        time.sleep(random.random()) # just in case if two processeses end at the same time tu to loading buffer
        featurizer = limit_featurizer(featurizer, datalength=datalength,dl_name=dl_name,ignore_existsing_feats=ignore_existsing_feats)
        featurizer["idx"] = np.arange(featurizer.shape[0]) + 1
    return


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
        #time.sleep(random.random()) # just in case if two processeses end at the same time tu to loading buffer
        featurizer = limit_featurizer(featurizer, datalength=datalength)
        featurizer["idx"] = np.arange(featurizer.shape[0]) + 1

    logger.info("nothing to do")



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataloader', type=str, required=True)
    parser.add_argument('--max_mols', type=int)
    parser.add_argument('-p', '--path', type=str, default=os.path.join(molNet.get_user_folder(), "autodata", "feats_raw"))
    args = parser.parse_args()
    main(dataloader=args.dataloader, max_mols=args.max_mols, path=args.path,ignore_existsing_feats=True,ignore_existsing_data=True,)
