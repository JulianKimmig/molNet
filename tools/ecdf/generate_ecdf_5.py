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

def load_mols(loader, limit=None) -> List[Mol]:
    if limit is not None and limit > 0:
        return loader.get_n_entries(limit)
    return [mol for mol in tqdm(
        loader, unit="mol", unit_scale=True, total=loader.expected_data_size, desc="load mols"
    )]


def retry_commit(session,dt=None,n=100):
    done=False
    for i in range(n-1):
        try:
            session.commit()
            done=True
            break
        except OperationalError as e:
            print(e)
            _dt=dt
            if _dt is None:
                _dt=np.random.rand()
            time.sleep(_dt)
            continue
    if not done:
        session.commit()


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

def raw_feat_mols(feat_row,mols,sql_data,dl_name,len_data=None, pos=None,ignore_existsing_data=False,inchies=None):
    if len_data is None:
        len_data = len(mols)
    feat = feat_row["instance"]
    text = f"{feat_row.name.rsplit('.', 1)[1]}"
    feat_row["DatasetsNumericalFeaturized"].working=True
    retry_commit(sql_data["session"])
    
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

    
    
    feature_table = Table(feat_row["db_entry"].name, sql_data["metadata"],
           Column('id', Integer, primary_key=True),
           *dp_cols
         )
    
    class _Feat(object):
        def __init__(self,**kwargs):
            for k,v in kwargs.items():
                setattr(self,k,v)
                
    mapper(_Feat, feature_table)
    sql_data["metadata"].create_all(sql_data["engine"])
    
    datalaoder,_=get_or_create(sql_data["session"], Dataset, dict(name=dl_name))


    #alle zuordungen von inchi + featurizer zu eintrag in der _Feat tabelle
    feat_inchi_keys_query = sql_data["session"].query(InchieKeyFeatures).filter_by(featurizer=feat_row["db_entry"])
    #alle inchies die da drin sind sollten exisieren
    is_done_inchie_keys = set([v.inchikey_id for v in feat_inchi_keys_query.all()])

    # alle inchikeys aus dem dataset
    key_query = sql_data["session"].query(InchieKeyPositionInDataset).filter_by(dataset=datalaoder)

    #alle positionen im dataset zu denen bereits ein inchie existiert
    in_positions=  {v.position for v in key_query.all()}

    # alle inchikeys aus dem dataset die schon gefeatured wurden
    res=key_query.filter(InchieKeyPositionInDataset.inchikey_id.in_(is_done_inchie_keys))
    #deren position im dataset
    is_done_position = set([r.position for r in  res.all()])


    #get_all needed inchikey ids to position
    inchikey_ids={v.position:v.inchie_key.id for v in key_query.options(joinedload(InchieKeyPositionInDataset.inchie_key)).all()}

    # fill unloaded inchies
    for i, mol in tqdm(enumerate(mols), desc=text, total=len_data, position=pos):
        if i in is_done_position:
            continue
        if i in inchikey_ids:
            continue
        #wenn noch kein Eintrag im dataset
        if i not in in_positions:
            inchikey,_ = get_or_create(sql_data["session"], InchieKey, {"key":MolToInchiKey(mol)},commit=True)
            ink_ds,_ = get_or_create(sql_data["session"], InchieKeyPositionInDataset, {"inchie_key":inchikey,"dataset":datalaoder,"position":i},commit=False)
            in_positions[i]=inchikey
        #normally this should not be called since its already done in gneration of inchikey_ids
        else:
            inchikey = key_query.filter_by(position=i).first().inchie_key

        inchikey_ids[i]=inchikey.id
    retry_commit(sql_data["session"])


    fds_id = feat_row["db_entry"].id
    def submit_new(new):
        retry_commit(sql_data["session"])
        for k,v in new.items():
            rentry,iks=v
            for ik_id in iks:
                fentry,_=get_or_create(sql_data["session"], InchieKeyFeatures,
                                         dict(
                                             inchikey_id=ik_id,
                                             featurizer_id=fds_id,
                                             feature_entry_id=rentry.id
                                         ),
                                         commit=False)
        retry_commit(sql_data["session"])

    new={}
    _in=0
    e_in_new=0
    finished=False
    try:
        for i, mol in tqdm(enumerate(mols), desc=text, total=len_data, position=pos):
            if i in is_done_position:
                _in+=1
                continue
            try:
                r = feat(mol)
                k=tuple(r.flatten().tolist())
                if k in new:
                    new[k][1].append(inchikey_ids[i])
                else:
                    rentry,_=get_or_create(sql_data["session"], _Feat, dict(zip(col_val_names,k)),commit=False)
                    new[k]=[rentry,[inchikey_ids[i]]]
                e_in_new+=1
                _in+=1

                if e_in_new>10000:
                    submit_new(new)
                    new={}
                    e_in_new=0

            except (molNet.ConformerError, ValueError, ZeroDivisionError) as e:
                print(e)
        finished=True
    except Exception as e:
        print(e)
    # shutil.rmtree(path)
    finally:
        submit_new(new)
        feat_row["DatasetsNumericalFeaturized"].working=False
        feat_row["DatasetsNumericalFeaturized"].finished=finished
        feat_row["DatasetsNumericalFeaturized"].size=_in
        retry_commit(sql_data["session"])
    return True


    

from sqlalchemy import Table,Column, Integer, String,Float,Boolean, ForeignKey, PrimaryKeyConstraint
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, mapper, load_only, declared_attr, joinedload

Base = declarative_base()

class Dataset(Base):
    __tablename__ = "datasets"
    id = Column(Integer, primary_key=True)
    name = Column(String)

class NumericalFeaturizer(Base):
    __tablename__ = "numerical_featurizer"
    id = Column(Integer, primary_key=True)
    name = Column(String)
    dim = Column(Integer, nullable=False)
    shape = Column(String, nullable=False)
    type = Column(String(1), nullable=False)

class DatasetsFeaturizedMixin():
    working = Column(Boolean,default=False)
    size = Column(Integer,default=0, nullable=False)
    finished = Column(Boolean,default=False)

    @declared_attr
    def dataset_id(cls):
        return Column('dataset_id', ForeignKey(Dataset.id))

    @declared_attr
    def featurizer_id(cls):
        return Column('featurizer_id', ForeignKey(cls._feat_class.id))

    @declared_attr
    def dataset(cls):
        return relationship(Dataset, foreign_keys=cls.dataset_id)

    @declared_attr
    def featurizer(cls):
        return relationship(cls._feat_class, foreign_keys=cls.featurizer_id)

    @declared_attr
    def __table_args__(cls):
        return (PrimaryKeyConstraint('dataset_id', 'featurizer_id'),)

class DatasetsNumericalFeaturized(DatasetsFeaturizedMixin,Base):
    __tablename__ = "datasets_numerical_featurized"
    _feat_class = NumericalFeaturizer


class InchieKey(Base):
    __tablename__="inchie_keys"
    id = Column(Integer, primary_key=True)
    key = Column(String)

class InchieKeyFeatures(Base):
    __tablename__="inchie_key_features"
    __table_args__ = (
        PrimaryKeyConstraint('inchikey_id', 'featurizer_id','feature_entry_id'),
    )
    inchikey_id = Column(Integer,ForeignKey(InchieKey.id))
    featurizer_id =Column(Integer,ForeignKey(NumericalFeaturizer.id))
    feature_entry_id = Column(Integer, nullable=False)
    
    inchie_key = relationship('InchieKey', foreign_keys='InchieKeyFeatures.inchikey_id')
    featurizer = relationship(NumericalFeaturizer, foreign_keys='InchieKeyFeatures.featurizer_id')

class InchieKeyPositionInDataset(Base):
    __table_args__ = (
        PrimaryKeyConstraint('inchikey_id', 'dataset_id','position'),
    )
    __tablename__ = "inchie_keys_pos_in_dataset"
    inchikey_id = Column(Integer,ForeignKey(InchieKey.id))
    dataset_id = Column(Integer, ForeignKey(Dataset.id))
    position=Column(Integer,nullable=False)
    
    inchie_key = relationship(InchieKey, foreign_keys='InchieKeyPositionInDataset.inchikey_id')
    dataset = relationship(Dataset, foreign_keys='InchieKeyPositionInDataset.dataset_id')


def get_or_create(session, model, filter_dict,commit=True):
    instance = session.query(model).filter_by(**filter_dict).first()
    if instance:
        return instance,False
    else:
        instance = model(**filter_dict)
        session.add(instance)
        if commit:
            retry_commit(session)
        return instance,True
    
from molNet.featurizer.molecule_featurizer import prepare_mol_for_featurization
SAMPLE_MOL=prepare_mol_for_featurization(MolFromSmiles("c1ccccc1"))

def ini_limit_featurizer(featurizer):
    logger.info(f"feats initial length = {len(featurizer)}")

    featurizer["isListFeat"] = featurizer["instance"].apply(lambda f: isinstance(f, FeaturizerList))
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

def limit_featurizer(featurizer, sql_data,datalength,ignore_existsing_feats=True,ignore_working_ds=True):
    retry_commit(sql_data["session"])
    if ignore_working_ds:
        featurizer["isworking"]=featurizer["DatasetsNumericalFeaturized"].apply(lambda _d: _d.working)
        featurizer.drop(featurizer.index[featurizer["isworking"]], inplace=True)

        logger.info(f"featurizer length after working = {len(featurizer)}")
        
    if ignore_existsing_feats:
        featurizer["isin"]=featurizer["DatasetsNumericalFeaturized"].apply(lambda _d: (_d.finished) & (_d.size>=(datalength*0.9)))
        featurizer.drop(featurizer.index[featurizer["isin"]], inplace=True)
        logger.info(f"featurizer length after existing = {len(featurizer)}")
    
    return featurizer


def create_tables(sql_data):
    Dataset.__table__.create(bind=sql_data["engine"], checkfirst=True)
    DatasetsNumericalFeaturized.__table__.create(bind=sql_data["engine"], checkfirst=True)
    NumericalFeaturizer.__table__.create(bind=sql_data["engine"], checkfirst=True)
    InchieKey.__table__.create(bind=sql_data["engine"], checkfirst=True)
    InchieKeyPositionInDataset.__table__.create(bind=sql_data["engine"], checkfirst=True)
    InchieKeyFeatures .__table__.create(bind=sql_data["engine"], checkfirst=True)

def create_featurizer_entries(featurizer,sql_data):
    def gen_feat_db_entry(r):
        f=r["instance"](SAMPLE_MOL)
        assert f.size == len(r["instance"])
        nf, _ = get_or_create(sql_data["session"],NumericalFeaturizer,dict(name=r.name,dim=f.ndim,shape=json.dumps(list(f.shape)),type=f.dtype.char),commit=False)
        return nf
    featurizer["db_entry"] = featurizer.apply(gen_feat_db_entry,axis=1)
    retry_commit(sql_data["session"])

    def gen_DatasetsNumericalFeaturized(r):
        nf, _ = get_or_create(sql_data["session"],DatasetsNumericalFeaturized,dict(dataset=sql_data["dl_table"],featurizer=r["db_entry"]),commit=False)
        return nf
    featurizer["DatasetsNumericalFeaturized"]=featurizer.apply(gen_DatasetsNumericalFeaturized,axis=1)

def get_and_create_inchies(sql_data,mols):
    key_query = sql_data["session"].query(InchieKeyPositionInDataset).filter_by(dataset=sql_data["dl_table"])
    iks={}
    multimols=set()
    if (key_query.count()/len(mols)) <0.9:
        for i, mol in tqdm(enumerate(mols), desc="load inchies", total=len(mols)):
            ik=MolToInchiKey(mol)
            inchikey,new= get_or_create(sql_data["session"], InchieKey, {"key":ik},commit=False)
            if new:
                iks[ik]=[(i,mol)]
            else:
                iks[ik].append((i,mol))
                multimols.add(ik)
            pos,_ = get_or_create(sql_data["session"], InchieKeyPositionInDataset, {"inchie_key":inchikey,"dataset":sql_data["dl_table"],"position":i},commit=False)
        retry_commit(sql_data["session"])
    for ik in multimols:
        print(ik)
        for (i,m) in iks[ik]:
            print("\t",i,m.GetProp("_Name"),MolToSmiles(m))

def main(dataloader, db, max_mols=None,ignore_existsing_feats=True,ignore_existsing_data=True,):
    if dataloader == "ChemBLdb29":
        from molNet.dataloader.molecular.ChEMBLdb import ChemBLdb29 as dataloaderclass
    elif dataloader == "ESOL":
        from molNet.dataloader.molecular.ESOL import ESOL as dataloaderclass
    else:
        raise ValueError(f"unknown dataloader '{dataloader}'")
        
    dl_name=f"{dataloaderclass.__module__}.{dataloaderclass.__name__}"
    logger.info(f"using db '{db}'")
    
    metadata = sql.MetaData() 
    engine = sql.create_engine('sqlite:///'+os.path.abspath(db), echo=False) 
    metadata.create_all(engine)
    Session = sessionmaker(engine)

    sql_data={
        "engine":engine,
        "metadata":metadata
    }

    with engine.connect() as conn:
        sql_data["connection"]=conn
        create_tables(sql_data)
        with Session(bind=conn) as session:
            sql_data["session"]=session

            dl_table,_= get_or_create(session, Dataset, dict(name=dl_name))
            sql_data["dl_table"]=dl_table
    
            logger.info("load mols")
            loader = PreparedMolDataLoader(dataloaderclass(
                data_streamer_kwargs=dict(iter_None=True))
            )
            mols = MolLoader(loader, limit=max_mols)
            get_and_create_inchies(sql_data,mols)


            # for mols
            featurizer = molNet.featurizer.get_molecule_featurizer_info()
            logger.info(f"limit featurizer for {dl_name}")
            featurizer = ini_limit_featurizer(featurizer)

            logger.info(f"generate featurizer entries")
            create_featurizer_entries(featurizer,sql_data)

            datalength=len(mols)

            while len(featurizer)>0:
                try:
                    time.sleep(random.random())
                    featurizer = limit_featurizer(featurizer,sql_data,datalength=datalength,ignore_existsing_feats=ignore_existsing_feats)
                    featurizer["idx"] = np.arange(featurizer.shape[0]) + 1
                    if len(featurizer)<=0:
                        break
                    logger.info(f"featurize {featurizer.iloc[0].name}")
                    r = raw_feat_mols(featurizer.iloc[0],mols,sql_data,dl_name,ignore_existsing_data=ignore_existsing_data)
                    if r:
                        featurizer.drop(featurizer.index[0],inplace=True)
                except OperationalError as e:
                    print(e)

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
