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


ignore=["GETAWAY_Featurizer",
        #"FpDensityMorgan1_Featurizer",
       ]

print("get mol file list")
loader = ecdf_conf.MOL_DATALOADER(ecdf_conf.MOL_DIR)
loader.data_streamer.cached=True
loader.get_n_entries(1000, progress_bar=True)

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


exit(0)



def _single_call_parallel_featurize_molgraph(d:List[MoleculeFeaturizer]):
    feats=d
    r=[]
    for f in feats:
        f.preferred_norm=None
        r.append(f(test_mol))
    return r



generated_test_feats = parallelize(
    _single_call_parallel_featurize_molgraph,
    molfeats,
    cores="all-1",
    progess_bar=True,
    progress_bar_kwargs=dict(unit=" feats"),
)


was_rem=True
if was_rem:
    removed=[]
    was_rem=False
    mol_files = [os.path.join(ecdf_conf.MOL_DIR, f) for f in os.listdir(ecdf_conf.MOL_DIR) if f.endswith(".mol")]
    print(f"found {len(mol_files)} mol files")
    nmf=10**int(np.log10(len(mol_files))-2) # eg6.4        
    nmf=int(np.floor(len(mol_files)/nmf)*nmf)
    print(nmf)
    
   # mol_files=mol_files[:1_000_000]

    for f in tqdm(mol_files,total=len(mol_files)):
        try:
            with open(f, "rb") as file:
                mol = Mol(file.read())
        except:
            os.remove(f)
            was_rem=True
            removed.append(f)
    
    for fr in removed:
        mol_files.remove(fr)
    if was_rem:
        print("not all mols could be read,redo")

nmf=10**int(np.log10(len(mol_files))-1) # eg6.4        
nmf=int(np.floor(len(mol_files)/nmf)*nmf)
mol_files=mol_files[:nmf]
print(f"read {len(mol_files)} mol files")


datadir = os.path.join(ecdf_conf.DATADIR,str(len(mol_files)))
import molNet.featurizer.molecule_featurizer as mf

molfeats=mf._available_featurizer
print(f"found {len(molfeats)} molecule featurizer")

# bool is already between 0 and 1
molfeats= [f for f in molfeats if f.dtype!=bool]
print(f"{len(molfeats)} remain after removal of bool types")

test_mol = MolFromSmiles("CCC")

def _single_call_parallel_featurize_molgraph(d:List[MoleculeFeaturizer]):
    feats=d
    r=[]
    for f in feats:
        f.preferred_norm=None
        r.append(f(test_mol))
    return r



generated_test_feats = parallelize(
    _single_call_parallel_featurize_molgraph,
    molfeats,
    cores="all-1",
    progess_bar=True,
    progress_bar_kwargs=dict(unit=" feats"),
)

molfeats=[molfeats[i] for i in range(len(molfeats)) if np.issubdtype(generated_test_feats[i].dtype,np.number)]
print(f"{len(molfeats)} remain after removal invalid types")



for f in molfeats:
    print(f"gen info {f}")
    f.ddir=os.path.join(datadir, inspect.getfile(f.__class__).replace(basedir_featurizer + os.sep, "").replace(".py", ""))
    os.makedirs(f.ddir,exist_ok=True)
    target_file=os.path.join(
        f.ddir,
        f"{f.__class__.__name__}_feature_info.pckl"
    )
    
    feature_info={}
    if os.path.exists(target_file):
        with open(target_file,"rb") as dfile:
            feature_info = pickle.load(dfile)
    
    feature_info["shape"]=f(test_mol).shape
    with open(target_file,"w+b") as dfile:
        pickle.dump(feature_info,dfile)


basedir_featurizer=os.path.dirname(inspect.getfile(MoleculeFeaturizer))

def _single_call_parallel_featurize_molfiles(d:Tuple[str,MoleculeFeaturizer]):
    feat=d[0][1]
    r = np.zeros((len(d),*test_mg.featurize_mol(feat).shape)) * np.nan
    for i, data in enumerate(d):
        try:
            with open(data[0], "rb") as file:
                mol = Mol(file.read())
        except:
            os.remove(file)
            continue
        feat=data[1]
        feat.preferred_norm=None
        try:
            r[i] =feat(mol)
        except (ConformerError, ValueError,ZeroDivisionError):
            pass
    return r

for f in molfeats:
    print(f"load {f}")
    ddir=os.path.join(datadir, inspect.getfile(f.__class__).replace(basedir_featurizer + os.sep, "").replace(".py", ""))
    os.makedirs(ddir,exist_ok=True)
    target_file=os.path.join(
        ddir,
        f"{f.__class__.__name__}_feature_dist.pckl"
    )
    if isinstance(f,VarSizeMoleculeFeaturizer):
        if os.path.exists(target_file):
            print("remove",target_file)
            #os.remove(target_file)
        continue
    if os.path.exists(target_file):
        continue
    
    if str(f) in ignore:
        print("ignore")
        continue
    
    mol_feats = parallelize(
        _single_call_parallel_featurize_molfiles,
        [(mf,f) for mf in mol_files],
        cores="all-1",
        progess_bar=True,
        progress_bar_kwargs=dict(unit=" feats"),
        split_parts=1000
    )

    
    #print(mol_feats)
    #for k in mol_feats:
    #    print(k.shape)
    with open(target_file,"w+b") as dfile:
        pickle.dump(mol_feats,dfile)
#pprint(dict(zip([str(f) for f in  molfeats],[(r,r.shape,r.min(),r.max()) for r in generated_test_feats])))
#f = m()(testmol)
#f=f+1
#assert np.issubdtype(f.dtype,np.number),f.dtype
#featurizer.append(m)
