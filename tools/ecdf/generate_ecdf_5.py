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

def raw_feat_mols(feat_row,mols,metadata,engine,session,dl_name,len_data=None, pos=None,ignore_existsing_data=False,inchies=None):
    if len_data is None:
        len_data = len(mols)
    feat = feat_row["instance"]
    text = f"{feat_row.name.rsplit('.', 1)[1]}"
    dsf = feat_row["DatasetsFeaturized"].working=True
    session.commit()
    
    if np.issubdtype(feat_row["dtype"], bool):
        DT=Boolean
    elif np.issubdtype(feat_row["dtype"], np.integer):
        DT=Integer
    elif np.issubdtype(feat_row["dtype"], np.floating):
        DT=Float
    else:
        raise ValueError(f"dtype not foud '{feat_row['dtype']}'")

    shape = json.loads(feat_row["db_entry"].shape)
    za=np.zeros(shape,dtype=feat_row["dtype"])
    
    col_val_names=[f'v_{i}' for i in range(len(za.flatten()))]
    dp_cols=[ Column(v, DT) for v in col_val_names]

    
    
    feature_table = Table(feat_row["db_entry"].name, metadata,
           Column('id', Integer, primary_key=True),
           *dp_cols
         )
    
    class _Feat(object):
        def __init__(self,**kwargs):
            for k,v in kwargs.items():
                setattr(self,k,v)
                
    mapper(_Feat, feature_table) 
    metadata.create_all(engine)
    
    datalaoder=get_or_create(session,Datasets,dict(name=dl_name))
    
   
    key_query = session.query(InchieKeysDataset).filter_by(dataset=datalaoder)
   
    #alle zuordungen von inchi + featurizer zu eintrag in der _Feat tabelle
    feat_inchi_keys_query = session.query(InchieKeysFeatures).filter_by(featurizer=feat_row["db_entry"])
    
    
    #alle inchies die da drin sind sollten exisieren
    is_done_inchie_keys = set([v.inchie_key.id for v in feat_inchi_keys_query.all()])

    
    res=key_query.filter(InchieKeysDataset.ik_id.in_(is_done_inchie_keys))
    is_done_position = set([r.position for r in  res.all()])
   

    
    
    in_positions=set([v.position for v in key_query.all()])
    new={}
    e_in_new=0
    fds_id = feat_row["db_entry"].fid
    
    def submit_new(new):
        session.commit()
        for k,v in new.items():
            rentry,iks=v
            for ik_id in iks:
                fentry=get_or_create(session, InchieKeysFeatures, dict(ik_id=ik_id,fds_id=fds_id,fentry_id=rentry.id),commit=False)
        session.commit()
    _in=0
    try:
        for i, mol in tqdm(enumerate(mols), desc=text, total=len_data, position=pos):
            if i in is_done_position:
                _in+=1
                continue
            if i not in in_positions:
                inchikey = get_or_create(session, InchieKeys, {"key":MolToInchiKey(mol)},commit=True)
                ink_ds = get_or_create(session, InchieKeysDataset, {"inchie_key":inchikey,"dataset":datalaoder,"position":i},commit=False)
            else:
                ink_ds = key_query.filter_by(position=i).first()
                inchikey = ink_ds.inchie_key
            try:
               
                r = feat(mol)
           
                k=tuple(r.flatten().tolist())
                if k in new:
                    new[k][1].append(inchikey.id)
                else:
                    rentry=get_or_create(session, _Feat, dict(zip(col_val_names,k)),commit=False)
                    new[k]=[rentry,[inchikey.id]]
                e_in_new+=1
                _in+=1
                
                if e_in_new>10000:
                    submit_new(new)
                    new={}
                    e_in_new=0
                
                #session.commit()
                #fentry=get_or_create(session, InchieKeysFeatures, dict(ik_id=inchikey.id,fds_id=feat_row["db_entry"].fid,fentry_id=rentry.id),commit=False)
            except (molNet.ConformerError, ValueError, ZeroDivisionError) as e:
                print(e)
        dsf = feat_row["DatasetsFeaturized"].finished=True
    except Exception as e:
        print(e)
       # shutil.rmtree(path)
    finally:
        submit_new(new)
        dsf = feat_row["DatasetsFeaturized"].working=False
        dsf = feat_row["DatasetsFeaturized"].size=_in
        session.commit()
        
        
    
    return
    
    
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
    

from sqlalchemy import Table,Column, Integer, String,Float,Boolean, ForeignKey, PrimaryKeyConstraint
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, mapper, load_only

Base = declarative_base()

class Datasets(Base):
    __tablename__ = "datasets"
    ds_id = Column(Integer, primary_key=True)
    name = Column(String)
    
class DatasetsFeaturized(Base):
    __table_args__ = (
        PrimaryKeyConstraint('dsfd_id', 'fds_id'),
    )
    
    __tablename__ = "datasets_featurized"
    dsfd_id = Column(Integer,ForeignKey('datasets.ds_id'))
    fds_id = Column(Integer,ForeignKey('numerical_feature.fid'))
    working = Column(Boolean,default=False)
    size = Column(Integer,default=0, nullable=False)
    finished = Column(Boolean,default=False)
    
    dataset = relationship('Datasets', foreign_keys='DatasetsFeaturized.dsfd_id')
    featurizer = relationship('NumericalFeaturizer', foreign_keys='DatasetsFeaturized.fds_id')
    
class NumericalFeaturizer(Base):
    __tablename__ = "numerical_feature"
    fid = Column(Integer, primary_key=True)
    name = Column(String)
    dim = Column(Integer, nullable=False)
    shape = Column(String, nullable=False)
    type = Column(String(1), nullable=False)

class InchieKeys(Base):
    __tablename__ = "inchie_keys"
    id = Column(Integer, primary_key=True)
    key = Column(String)

class InchieKeysFeatures(Base):
    __table_args__ = (
        PrimaryKeyConstraint('ik_id', 'fds_id','fentry_id'),
    )
    __tablename__ = "inchie_keys_feature"
    ik_id = Column(Integer,ForeignKey('inchie_keys.id'))
    fds_id =Column(Integer,ForeignKey('numerical_feature.fid'))
    fentry_id = Column(Integer, nullable=False)
    
    inchie_key = relationship('InchieKeys', foreign_keys='InchieKeysFeatures.ik_id')
    featurizer = relationship('NumericalFeaturizer', foreign_keys='InchieKeysFeatures.fds_id')

class InchieKeysDataset(Base):
    __table_args__ = (
        PrimaryKeyConstraint('ik_id', 'dsfd_id','position'),
    )
    __tablename__ = "inchie_keys_dataset"
    ik_id = Column(Integer,ForeignKey('inchie_keys.id'))
    dsfd_id = Column(Integer,ForeignKey('datasets.ds_id'))
    position=Column(Integer,nullable=False)
    
    inchie_key = relationship('InchieKeys', foreign_keys='InchieKeysDataset.ik_id')
    dataset = relationship('Datasets', foreign_keys='InchieKeysDataset.dsfd_id')
    
    
def get_or_create(session, model, filter_dict,commit=True):
    instance = session.query(model).filter_by(**filter_dict).first()
    if instance:
        return instance
    else:
        instance = model(**filter_dict)
        session.add(instance)
        if commit:
            session.commit()
        return instance
    
from molNet.featurizer.molecule_featurizer import prepare_mol_for_featurization
SAMPLE_MOL=prepare_mol_for_featurization(MolFromSmiles("c1ccccc1"))
def limit_featurizer(featurizer, dl_name,session,ignore_existsing_feats=True,ignore_working_ds=True):
    
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
    
    datalaoder=get_or_create(session,Datasets,dict(name=dl_name))
    
    def _dt(r):
        return np.issubdtype(r["dtype"],np.number)
    
    featurizer = featurizer[featurizer.apply(_dt,axis=1)]
    logger.info(f"featurizer length after non numeric drop = {len(featurizer)}")
    
    def gen_feat_db_entry(r):
        f=r["instance"](SAMPLE_MOL)
        return get_or_create(session,NumericalFeaturizer,dict(name=r.name,dim=f.ndim,shape=json.dumps(list(f.shape)),type=f.dtype.char),commit=False)
    featurizer["db_entry"] = featurizer.apply(gen_feat_db_entry,axis=1)
    session.commit()
    
    def gen_DatasetsFeaturized(r):
        return get_or_create(session,DatasetsFeaturized,dict(dataset=datalaoder,featurizer=r["db_entry"]),commit=False)
    featurizer["DatasetsFeaturized"]=featurizer.apply(gen_DatasetsFeaturized,axis=1)
    
    if ignore_working_ds:
        featurizer["isworking"]=False
        session.commit()
        featurizer["isworking"]=featurizer["DatasetsFeaturized"].apply(lambda _d: _d.working)
        
        featurizer = featurizer[~featurizer["isworking"]]
        logger.info(f"featurizer length after working = {len(featurizer)}")
        
    if ignore_existsing_feats:
        featurizer["isin"]=False
        featurizer["DatasetsFeaturized"]=featurizer.apply(gen_DatasetsFeaturized,axis=1)
        session.commit()
        
        featurizer["isin"]=featurizer["DatasetsFeaturized"].apply(lambda _d: _d.finished)
        featurizer = featurizer[~featurizer["isin"]]
        logger.info(f"featurizer length after existing = {len(featurizer)}")
    
    return featurizer

def main(dataloader, db, max_mols=None,ignore_existsing_feats=True,ignore_existsing_data=True,):
    if dataloader == "ChemBLdb29":
        from molNet.dataloader.molecular.ChEMBLdb import ChemBLdb29 as dataloaderclass
    elif dataloader == "ESOL":
        from molNet.dataloader.molecular.ESOL import ESOL as dataloaderclass
    else:
        raise ValueError(f"unknown dataloader '{dataloader}'")
        
    dl_name=f"{dataloaderclass.__module__}.{dataloaderclass.__name__}"
    
    
    
    metadata = sql.MetaData() 
    engine = sql.create_engine('sqlite:///'+os.path.abspath(db), echo=False) 
    metadata.create_all(engine)
    Session = sessionmaker(engine)
    
    with engine.connect() as conn:

        Datasets.__table__.create(bind=engine, checkfirst=True)
        DatasetsFeaturized.__table__.create(bind=engine, checkfirst=True)
        NumericalFeaturizer.__table__.create(bind=engine, checkfirst=True)
        InchieKeys.__table__.create(bind=engine, checkfirst=True)
        InchieKeysDataset.__table__.create(bind=engine, checkfirst=True)
        InchieKeysFeatures .__table__.create(bind=engine, checkfirst=True)
        
        with Session(bind=conn) as session:
            dl_table= get_or_create(session,Datasets,dict(name=dl_name))
        

    
    
            logger.info("load mols")
            loader = PreparedMolDataLoader(dataloaderclass())
            mols = MolLoader(loader, limit=max_mols)

            # for mols
            datalength=len(mols)
            featurizer = molNet.featurizer.get_molecule_featurizer_info()
            #dl_path = os.path.join(path, "raw_features", dataloader)

            #featurizer["path"] = [os.path.join(path, *mod.split(".")) for mod in featurizer.index]


            
            
            key_query = session.query(InchieKeysDataset).filter_by(dataset=dl_table)
            
            
            
            if (key_query.count()/len(mols)) <0.9:
                for i, mol in tqdm(enumerate(mols), desc="load inchies", total=datalength):
                    inchikey= get_or_create(session, InchieKeys, {"key":MolToInchiKey(mol)},commit=False)
                    get_or_create(session, InchieKeysDataset, {"inchie_key":inchikey,"dataset":dl_table,"position":i},commit=False)
                session.commit()
            
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
            
            featurizer = limit_featurizer(featurizer, dl_name=dl_name,session=session,ignore_existsing_feats=ignore_existsing_feats)
            featurizer["idx"] = np.arange(featurizer.shape[0]) + 1
            while len(featurizer)>0:
                logger.info(f"featurize {featurizer.iloc[0].name}")
          
                r = raw_feat_mols(featurizer.iloc[0],mols,metadata,engine,session,dl_name,ignore_existsing_data=ignore_existsing_data,inchies=inchies)
                
                if r:
                    featurizer.drop(featurizer.index[0],inplace=True)
                time.sleep(random.random()) # just in case if two processeses end at the same time tu to loading buffer
                featurizer = limit_featurizer(featurizer, dl_name=dl_name,session=session,ignore_existsing_feats=ignore_existsing_feats)
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
    parser.add_argument('--db', type=str, default=os.path.join(molNet.get_user_folder(), "autodata", "feats_raw.db"))
    args = parser.parse_args()
    main(dataloader=args.dataloader, max_mols=args.max_mols, db=args.db,ignore_existsing_feats=True,ignore_existsing_data=True,)
