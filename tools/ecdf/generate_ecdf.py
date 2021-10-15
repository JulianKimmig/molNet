import os
import sys

modp = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0,modp)
sys.path.append(modp)

from tools.ecdf import ecdf_conf

import inspect
import pickle

import numpy as np
from typing import List, Tuple

from rdkit import RDLogger
import sys

from tqdm import tqdm

from molNet import ConformerError
from molNet.featurizer._molecule_featurizer import MoleculeFeaturizer,VarSizeMoleculeFeaturizer
from molNet.utils.parallelization.multiprocessing import parallelize

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)
import logging
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger('matplotlib').setLevel(logging.WARNING)

from rdkit.Chem import Mol, MolFromSmiles
import json
import time

ignore=["GETAWAY_Featurizer",
        #"FpDensityMorgan1_Featurizer",
       ]

print("get mol file list")
loader = ecdf_conf.MOL_DATALOADER(ecdf_conf.MOL_DIR)


test_mol = MolFromSmiles("CCC")

def _single_call_parallel_featurize_molgraph(d:List[MoleculeFeaturizer]):
    feats=d
    r=[]
    for f in feats:
        f.preferred_norm=None
        r.append(f(test_mol))
    return r

import molNet.featurizer.molecule_featurizer as mf

molfeats=mf._available_featurizer
print(f"found {len(molfeats)} molecule featurizer")

# bool is already between 0 and 1
molfeats= [f for f in molfeats if f.dtype!=bool]
molfeats= [f for f in molfeats if f not in ignore]
print(f"{len(molfeats)} remain after removal of bool types")

test_mol = MolFromSmiles("CCC")

generated_test_feats = parallelize(
    _single_call_parallel_featurize_molgraph,
    molfeats,
    cores="all-1",
    progess_bar=True,
    progress_bar_kwargs=dict(unit=" feats"),
)


molfeats=[molfeats[i] for i in range(len(molfeats)) if np.issubdtype(generated_test_feats[i].dtype,np.number)]
print(f"{len(molfeats)} remain after removal invalid types")

basedir_featurizer=os.path.dirname(inspect.getfile(MoleculeFeaturizer))

for f in molfeats:
    print(f"gen info {f}")
    f.ddir=os.path.join(ecdf_conf.DATADIR, inspect.getfile(f.__class__).replace(basedir_featurizer + os.sep, "").replace(".py", ""))
    os.makedirs(f.ddir,exist_ok=True)
    target_file=os.path.join(
        f.ddir,
        f"{f.__class__.__name__}_feature_info.json"
    )
    
    feature_info={}
    if os.path.exists(target_file):
        with open(target_file,"r") as dfile:
            feature_info = json.load(dfile)
    
    feature_info["shape"]=f(test_mol).shape
    with open(target_file,"w+") as dfile:
        json.dump(feature_info,dfile,indent=4)




def _single_call_parallel_featurize_molfiles(d:Tuple[Mol,MoleculeFeaturizer]):
    feat=d[0][1]
    r = np.zeros((len(d),*feat(test_mol).shape)) * np.nan
    for i, data in enumerate(d):
        mol = data[0]
        feat=data[1]
        feat.preferred_norm=None
        try:
            r[i] =feat(mol)
        except (ConformerError, ValueError,ZeroDivisionError):
            pass
    return r

loader = ecdf_conf.MOL_DATALOADER(ecdf_conf.MOL_DIR)
mols=[mol for mol in tqdm(
        loader, unit="mol", unit_scale=True, total=loader.expected_data_size
    )]

for f in molfeats:
    print(f"load {f}")
    ddir=os.path.join(ecdf_conf.DATADIR, inspect.getfile(f.__class__).replace(basedir_featurizer + os.sep, "").replace(".py", ""))
    os.makedirs(ddir,exist_ok=True)
    target_file=os.path.join(
        ddir,
        f"{f.__class__.__name__}_feature_dist.pckl"
    )
    if isinstance(f,VarSizeMoleculeFeaturizer):
        if os.path.exists(target_file):
            print("remove",target_file)
            os.remove(target_file)
        continue
    if os.path.exists(target_file):
        continue
    
    if str(f) in ignore:
        print("ignore")
        continue
    ts=time.time()
    mol_feats = parallelize(
        _single_call_parallel_featurize_molfiles,
        [(mf,f) for mf in mols],
        cores="all-1",
        progess_bar=True,
        progress_bar_kwargs=dict(unit=" feats"),
        split_parts=1000
    )
    te=time.time()
    
    with open(target_file,"w+b") as dfile:
        pickle.dump(mol_feats,dfile)
    
    
    
    target_file=os.path.join(
        f.ddir,
        f"{f.__class__.__name__}_feature_info.json"
    )
    feature_info={}
    if os.path.exists(target_file):
        with open(target_file,"r") as dfile:
            feature_info = json.load(dfile)
    
    feature_info["time"]=te-ts
    with open(target_file,"w+") as dfile:
        json.dump(feature_info,dfile,indent=4)

