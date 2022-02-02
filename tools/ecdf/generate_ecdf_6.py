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


    

from sqlalchemy import Table, Column, Integer, String, Float, Boolean, ForeignKey, PrimaryKeyConstraint, \
    UniqueConstraint
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, mapper, load_only, declared_attr, joinedload, Query

Base = declarative_base()
#
# class NumericalFeaturizer(Base):
#     __tablename__ = "numerical_featurizer"
#     id = Column(Integer, primary_key=True)
#     name = Column(String)
#     dim = Column(Integer, nullable=False)
#     shape = Column(String, nullable=False)
#     type = Column(String(1), nullable=False)
#
# class DatasetsFeaturizedMixin():
#     working = Column(Boolean,default=False)
#     size = Column(Integer,default=0, nullable=False)
#     finished = Column(Boolean,default=False)
#
#     @declared_attr
#     def dataset_id(cls):
#         return Column('dataset_id', ForeignKey(Dataset.id))
#
#     @declared_attr
#     def featurizer_id(cls):
#         return Column('featurizer_id', ForeignKey(cls._feat_class.id))
#
#     @declared_attr
#     def dataset(cls):
#         return relationship(Dataset, foreign_keys=cls.dataset_id)
#
#     @declared_attr
#     def featurizer(cls):
#         return relationship(cls._feat_class, foreign_keys=cls.featurizer_id)
#
#     @declared_attr
#     def __table_args__(cls):
#         return (PrimaryKeyConstraint('dataset_id', 'featurizer_id'),)
#
# class DatasetsNumericalFeaturized(DatasetsFeaturizedMixin,Base):
#     __tablename__ = "datasets_numerical_featurized"
#     _feat_class = NumericalFeaturizer
#
#
# class InchieKey(Base):
#     __tablename__="inchie_keys"
#     id = Column(Integer, primary_key=True)
#     key = Column(String)
#
# class InchieKeyFeatures(Base):
#     __tablename__="inchie_key_features"
#     __table_args__ = (
#         PrimaryKeyConstraint('inchikey_id', 'featurizer_id','feature_entry_id'),
#     )
#     inchikey_id = Column(Integer,ForeignKey(InchieKey.id))
#     featurizer_id =Column(Integer,ForeignKey(NumericalFeaturizer.id))
#     feature_entry_id = Column(Integer, nullable=False)
#
#     inchie_key = relationship('InchieKey', foreign_keys='InchieKeyFeatures.inchikey_id')
#     featurizer = relationship(NumericalFeaturizer, foreign_keys='InchieKeyFeatures.featurizer_id')
#
# class InchieKeyPositionInDataset(Base):
#     __table_args__ = (
#         PrimaryKeyConstraint('inchikey_id', 'dataset_id','position'),
#     )
#     __tablename__ = "inchie_keys_pos_in_dataset"
#     inchikey_id = Column(Integer,ForeignKey(InchieKey.id))
#     dataset_id = Column(Integer, ForeignKey(Dataset.id))
#     position=Column(Integer,nullable=False)
#
#     inchie_key = relationship(InchieKey, foreign_keys='InchieKeyPositionInDataset.inchikey_id')
#     dataset = relationship(Dataset, foreign_keys='InchieKeyPositionInDataset.dataset_id')
#

def retry_session(sql_data,func,*args,n=1000,dt=None,**kwargs):
    done=False
    r=None
    for i in range(n-1):
        with sql_data["SessionMaker"]() as session:
            try:
                r= func(session,*args,**kwargs)
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
        with sql_data["SessionMaker"]() as session:
            r= func(session,*args,**kwargs)
            session.commit()
    return r



def get_or_create(session, model, filter_dict,commit=True):

    instance = session.query(model).filter_by(**filter_dict).first()

    if instance:
        return instance,False
    else:
        instance = model(**filter_dict)
        if commit:
            session.add(instance)
            session.commit()
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




def create_tables(sql_data):
    Dataset.__table__.create(bind=sql_data["engine"], checkfirst=True)
    DatasetsNumericalFeaturized.__table__.create(bind=sql_data["engine"], checkfirst=True)
    NumericalFeaturizer.__table__.create(bind=sql_data["engine"], checkfirst=True)
    InchieKey.__table__.create(bind=sql_data["engine"], checkfirst=True)
    InchieKeyPositionInDataset.__table__.create(bind=sql_data["engine"], checkfirst=True)
    InchieKeyFeatures .__table__.create(bind=sql_data["engine"], checkfirst=True)






class InchieValueLink(Base):
    __table_args__ = (
        UniqueConstraint('inchikey_id','value_id'),
    )
    __tablename__ = "inchie_value_link"
    inchikey_id = Column(Integer,nullable=False,primary_key=True)
    value_id = Column(Integer,nullable=False)

class Dataset(Base):
    __tablename__ = "datasets"
    id = Column(Integer, primary_key=True)
    name = Column(String,unique=True)

class Featurizer(Base):
    __tablename__ = "featurizer"
    id = Column(Integer, primary_key=True)
    name = Column(String,unique=True)
    shape=Column(String, nullable=False)

class InchieKey(Base):
    __tablename__="inchie_keys"
    id = Column(Integer, primary_key=True)
    key = Column(String,unique=True)

class InchieKeyPositionInDataset(Base):
     __table_args__ = (
         PrimaryKeyConstraint('inchikey_id','position'),
     )
     __tablename__ = "inchie_keys_pos_in_dataset"
     inchikey_id = Column(Integer,ForeignKey(InchieKey.id))
     position=Column(Integer,nullable=False,unique=True)

class DatasetFeaturizerStatus(Base):
    __tablename__ = "dataset_featurizer_status"
    featurizer_id = Column(Integer,nullable=False,unique=True,primary_key=True)
    working=Column(Boolean,nullable=False,default=False)
    finished=Column(Boolean,nullable=False,default=False)
    size=Column(Integer,nullable=False,default=0)


def create_db(path,*tabels):
    metadata = sql.MetaData()
    engine = sql.create_engine('sqlite:///'+path, echo=False)
    metadata.create_all(engine)
    for t in tabels:
        t.__table__.create(bind=engine, checkfirst=True)
    return engine

def create_featurizer_entries(featurizer,dataset_featurizer_engine):
    featurizer["featurizer_id"]=None
    with sessionmaker(dataset_featurizer_engine)() as session:
        known_featurizer=[
                (f.name,f.id)
                for f in session.query(Featurizer).filter(
                Featurizer.name.in_(
                    set(featurizer["featurizer_id"].index)
                )
            ).all()
            ]

    known_featurizer_ids = [f[1] for f in known_featurizer]
    known_featurizer_names = [f[0] for f in known_featurizer]

    new_featurizer= set(featurizer["featurizer_id"].index)-set(known_featurizer_names)

    for n in new_featurizer:
        r=featurizer.loc[n]
        f=r["instance"](SAMPLE_MOL)
        assert f.size == len(r["instance"])
        featurizer.loc[n,"shape"]=json.dumps(f.shape)

    new_featurizer=[Featurizer(name=n,shape=featurizer.loc[n,"shape"]) for n in new_featurizer]
    with sessionmaker(dataset_featurizer_engine)() as session:
        session.bulk_save_objects(new_featurizer)
        session.commit()
    with sessionmaker(dataset_featurizer_engine)() as session:
        all_featurizer=session.query(Featurizer).filter(
            Featurizer.name.in_(
                set(featurizer["featurizer_id"].index)
            )
        ).all()
    for f in all_featurizer:
        featurizer.loc[f.name,"featurizer_id"] = f.id
        featurizer.loc[f.name,"shape"] = f.id


def create_main_db(path):
    dataset_featurizer_engine = create_db(os.path.join(path,"dataset_featurizer.db"),
                                          Dataset,Featurizer
                                          )

    inchikeys_engine = create_db(os.path.join(path,"inchikeys.db"),
                                 InchieKey
                                 )

    return dataset_featurizer_engine,inchikeys_engine

def create_dataset_db(path,id):
    dataset_engine = create_db(os.path.join(path,f"dataset_{id}.db"),
                               InchieKeyPositionInDataset,
                               DatasetFeaturizerStatus
                               )
    return dataset_engine

def create_featurizer_db(path,feat_row):
    if np.issubdtype(feat_row["dtype"], bool):
        DT=Boolean
    elif np.issubdtype(feat_row["dtype"], np.integer):
        DT=Integer
    elif np.issubdtype(feat_row["dtype"], np.floating):
        DT=Float
    else:
        raise ValueError(f"dtype not foud '{feat_row['dtype']}'")

    shape = json.loads(feat_row["shape"])
    za=np.zeros(shape,dtype=feat_row["dtype"])

    col_val_names=[f'v_{i}' for i in range(len(za.flatten()))]
    dp_cols=[ Column(v, DT) for v in col_val_names]

    metadata = sql.MetaData()
    feature_table = Table("featurizer_values",metadata,
                          Column('id', Integer, primary_key=True),
                          *dp_cols,UniqueConstraint(*col_val_names)
                          )
    class _Feat(object):
        __table_args__ = (
            PrimaryKeyConstraint(*col_val_names),
        )
        def __init__(self,**kwargs):
            for k,v in kwargs.items():
                setattr(self,k,v)

    mapper(_Feat, feature_table)

    featurizer_engine = create_db(os.path.join(path,f"featurizer_{feat_row['featurizer_id']}.db"),
                                  InchieValueLink,
                                  )

    metadata.create_all(featurizer_engine)


    return featurizer_engine,_Feat

def get_and_create_inchies(inchikeys_engine,dataset_engine,mols):
    d=100000
    complete=False
    positional_inchies=-np.ones(len(mols),dtype=int)
    while not complete:
        complete=True
        known_positions = set()
        logger.info("get known positions")
        with sessionmaker(dataset_engine)() as session:
            s=len(mols)
            k=0
            while k<s:
                data = session.query(InchieKeyPositionInDataset).filter(InchieKeyPositionInDataset.position.in_(range(k,min(k+d,s)))).all()
                positional_inchies[[p.position for p in data]]=[p.inchikey_id for p in data]
                known_positions.update(
                    set(
                        [p.position for p in data]
                    )
                )
                k=k+d

        keys=[]
        positions=[]
        n=0
        final_continuous=-1
        for i in range(len(known_positions)):
            if i in known_positions:
                final_continuous=i
                continue
            else:
                break

        if (len(known_positions)/len(mols)) <1:
            logger.info(f"create unknown inchies ({len(mols)-len(known_positions)})")
            for i, mol in tqdm(enumerate(mols), desc="load inchies", total=len(mols)):
                if i<=final_continuous or i in known_positions:
                    continue
                ik=MolToInchiKey(mol)
                keys.append(ik)
                positions.append(i)
                n+=1
                if n>=d:
                    complete=False
                    break

        unique_keys=set(keys)

        if len(unique_keys)>0:
            logger.info("save new inchies")
            with sessionmaker(inchikeys_engine)() as session:
                existing_keys = [k.key for k in session.query(InchieKey).filter(InchieKey.key.in_(unique_keys)).all()]

            for k in existing_keys:
                unique_keys.remove(k)

            existing_keys = [InchieKey(key=k) for k in unique_keys]

            with sessionmaker(inchikeys_engine)() as session:
                session.bulk_save_objects(existing_keys)
                session.commit()

        if len(keys)>0:
            logger.info("save new positions")
            with sessionmaker(inchikeys_engine)() as session:
                all_keys = {k.key:k.id for k in session.query(InchieKey).filter(InchieKey.key.in_(set(keys))).all()}

            new_pos=[InchieKeyPositionInDataset(position = positions[i],inchikey_id = all_keys[k]) for i,k in enumerate(keys)]

            with sessionmaker(dataset_engine)() as session:
                session.bulk_save_objects(new_pos)
                session.commit()
            positional_inchies[[p.position for p in new_pos]]=[p.inchikey_id for p in new_pos]
    return positional_inchies

def get_dataset_info(dataset_class,dataset_featurizer_engine):
    dl_name=f"{dataset_class.__module__}.{dataset_class.__name__}"
    with sessionmaker(dataset_featurizer_engine)() as session:
        dl_table,_= get_or_create(session, Dataset, dict(name=dl_name))
        dataset_id=dl_table.id
        dataset_name=dl_table.name
    return dataset_id,dataset_name

def update_featurizer_df(featurizer,dataset_featurizer_engine, dataset_engine):
    featurizer["working"]=False
    featurizer["finished"]=False
    featurizer["size"]=0
    featurizer["db_shape"]=""
    with sessionmaker(dataset_featurizer_engine)() as session:
        all_featurizer=session.query(Featurizer).filter(
            Featurizer.name.in_(
                set(featurizer["featurizer_id"].index)
            )
        ).all()

    for f in all_featurizer:
        featurizer.loc[f.name,"featurizer_id"] = f.id
        featurizer.loc[f.name,"shape"] = f.shape

    with sessionmaker(dataset_engine)() as session:
        all_featurizer=session.query(DatasetFeaturizerStatus).filter(
            DatasetFeaturizerStatus.featurizer_id.in_(
                set(featurizer["featurizer_id"])
            )
        ).all()

    found_indices=set()
    for f in all_featurizer:
        idx = featurizer[featurizer["featurizer_id"] == f.featurizer_id].index.values[0]
        found_indices.add(idx)
        featurizer.loc[idx,"working"] = f.finished
        featurizer.loc[idx,"finished"] = f.finished
        featurizer.loc[idx,"size"] = f.size

    new_stati=[]
    for r,d in featurizer.iterrows():
        if r not in found_indices:
            new_stati.append(
                DatasetFeaturizerStatus(
                    featurizer_id = d["featurizer_id"],
                working=False,
                finished=False,
                size=0
                )
            )
    if len(new_stati)>0:
        with sessionmaker(dataset_engine)() as session:
            session.bulk_save_objects(new_stati)
            session.commit()


def get_featurized_inchies(featurizer_engine):
    with sessionmaker(featurizer_engine)() as session:
        ichie_ids = [p.inchikey_id for p in session.query(InchieValueLink).all()]
    return ichie_ids


def get_next_featurizer(featurizer,dataset_featurizer_engine, dataset_engine,datalength,ignore_existsing_feats=True,ignore_working_ds=True):
    update_featurizer_df(featurizer=featurizer,
                     dataset_featurizer_engine=dataset_featurizer_engine,
                     dataset_engine=dataset_engine)

    idx = featurizer[(~featurizer["working"])&(~featurizer["finished"])].index[0]
    return featurizer.loc[idx]


def featurize_mols(feat_row, mols,positional_inchies,featurizer_engine,value_class, skip_positions=None):
    if skip_positions is None:
        skip_positions = []
    feat=feat_row["instance"]

    col_names = sorted([v for v in value_class.__dict__.keys() if v.startswith("v_")],key=lambda v: int(v[2:]))

    with sessionmaker(featurizer_engine)() as session:
        all_values = session.query(value_class).all()
        all_values = [(tuple([getattr(v,cn) for cn in col_names]),v.id) for v in all_values]

    known_value_ids=dict(all_values)

    known_values= {av[0]:set() for av in all_values}

    print(known_value_ids)
    print(known_values)

    new_values=dict()
    first_uneven=-1
    if len(skip_positions)>0:
        first_uneven=((np.arange(len(skip_positions))-np.array(skip_positions))!=0).argmax()
        if skip_positions[first_uneven] == 0:
            first_uneven = len(skip_positions)

    worked=0
    if first_uneven<len(mols):
        for i, mol in tqdm(enumerate(mols), desc="load inchies", total=len(mols)):
            worked+=1
            if i<first_uneven:
                continue
            if i in skip_positions:
                continue
            inchie_id=positional_inchies[i]
            _f=feat(mol)
            ff=tuple(_f.flatten())
            if ff in known_values:
                known_values[ff].add(inchie_id)
            else:
                if ff in new_values:
                    new_values[ff].append(inchie_id)
                else:
                    new_values[ff]=[inchie_id]

    new_values_entries=[
        value_class(**dict(zip(col_names,value)))
        for value,inchie_ids in new_values.items()
    ]

    if len(new_values_entries) > 0:
        with sessionmaker(featurizer_engine)() as session:
            session.bulk_save_objects(new_values_entries)
            session.commit()

    new_value_links=[]
    in_inchies=set()

    for value,inchie_ids in known_values.items():
        value_id = known_value_ids[value]
        for iid in inchie_ids:
            if iid in in_inchies:
                continue
            new_value_links.append(
                InchieValueLink(inchikey_id=int(iid),value_id=int(value_id))
            )
            in_inchies.add(iid)

    if len(new_values_entries) > 0:
        with sessionmaker(featurizer_engine)() as session:
            all_values = session.query(value_class).all()
            all_values = [(tuple([getattr(v,cn) for cn in col_names]),v.id) for v in all_values]
        known_value_ids=dict(all_values)

        for value,inchie_ids in new_values.items():
            value_id = known_value_ids[value]
            for iid in inchie_ids:
                if iid in in_inchies:
                    continue
                new_value_links.append(
                    InchieValueLink(inchikey_id=int(iid),value_id=int(value_id))
                )
                in_inchies.add(iid)

    if len(new_value_links)>0:
        with sessionmaker(featurizer_engine)() as session:
            session.bulk_save_objects(new_value_links)
            session.commit()

    return worked,True

def main(dataloader, path, max_mols=None,ignore_existsing_feats=True,ignore_existsing_data=True,):
    if dataloader == "ChemBLdb29":
        from molNet.dataloader.molecular.ChEMBLdb import ChemBLdb29 as dataloaderclass
    elif dataloader == "ESOL":
        from molNet.dataloader.molecular.ESOL import ESOL as dataloaderclass
    else:
        raise ValueError(f"unknown dataloader '{dataloader}'")

    path=os.path.abspath(path)
    logger.info(f"using path '{path}'")

    os.makedirs(path,exist_ok=True)
    dataset_featurizer_engine,inchikeys_engine = create_main_db(path)

    dataset_id,dataset_name = get_dataset_info(dataloaderclass,dataset_featurizer_engine)

    datasets_path=os.path.join(path,"datasets")
    featurizer_db_path=os.path.join(path,"featurizer")
    os.makedirs(datasets_path,exist_ok=True)
    os.makedirs(featurizer_db_path,exist_ok=True)

    dataset_engine = create_dataset_db(datasets_path,dataset_id)

    logger.info("load mols")
    loader = PreparedMolDataLoader(dataloaderclass(
        data_streamer_kwargs=dict(iter_None=True))
    )
    mols = MolLoader(loader, limit=max_mols)
    positional_inchies = get_and_create_inchies(inchikeys_engine,dataset_engine,mols)



    # for mols
    featurizer = molNet.featurizer.get_molecule_featurizer_info()
    logger.info(f"limit featurizer for {dataset_name}")
    featurizer = ini_limit_featurizer(featurizer)



    logger.info(f"generate featurizer entries")
    create_featurizer_entries(featurizer,dataset_featurizer_engine)


    datalength=len(mols)

    while len(featurizer)>0:
        try:
            time.sleep(random.random())
            feat_row = get_next_featurizer(featurizer,dataset_featurizer_engine,dataset_engine,datalength=datalength,ignore_existsing_feats=ignore_existsing_feats)
            with sessionmaker(dataset_engine)() as session:
                feat=session.query(DatasetFeaturizerStatus).filter_by(
                    featurizer_id=feat_row["featurizer_id"],
                ).first()
                feat.working=True
                session.commit()

            logger.info(f"featurize {feat_row.name}")
            featurizer_engine,featurizer_class = create_featurizer_db(featurizer_db_path,feat_row=feat_row)
            featurized_inchies = get_featurized_inchies(featurizer_engine)
            featurized_positions = np.where(np.in1d(positional_inchies, featurized_inchies))[0]
            worked,done = featurize_mols(feat_row,mols,value_class=featurizer_class,featurizer_engine=featurizer_engine,positional_inchies=positional_inchies,skip_positions=featurized_positions)

            with sessionmaker(dataset_engine)() as session:
                feat=session.query(DatasetFeaturizerStatus).filter_by(
                    featurizer_id=feat_row["featurizer_id"],
                ).first()
                feat.working=False
                feat.size=worked
                if done:
                    feat.finished=True
                session.commit()
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
    parser.add_argument('--path', type=str, default=os.path.join(molNet.get_user_folder(), "autodata", "feats_raw"))
    args = parser.parse_args()
    main(dataloader=args.dataloader, max_mols=args.max_mols, path=args.path,ignore_existsing_feats=True,ignore_existsing_data=True,)
