import json
import os
import pickle
import shutil
import string

import numpy as np
from tqdm import tqdm

from molNet import get_user_folder, MOLNET_LOGGER
from molNet.dataloader.dataloader import DataLoader
from molNet.dataloader.molecular.dataloader import MolDataLoader
from molNet.dataloader.molecular.streamer import PickledMolStreamer
from molNet.featurizer._atom_featurizer import FixedSizeAtomFeaturizer, _AtomFeaturizer
from molNet.featurizer._molecule_featurizer import FixedSizeMoleculeFeaturizer, prepare_mol_for_featurization, \
    _MoleculeFeaturizer
from molNet.featurizer.featurizer import Featurizer, FixedSizeFeaturizer

valid_chars = "-_.(), %s%s" % (string.ascii_letters, string.digits)

def molecule_fixed_size_featuizer_handler(featurizer:FixedSizeMoleculeFeaturizer,dataloader:MolDataLoader,data_path,n_split=100_000):
    lenmols=dataloader.expected_data_size
    featurizer_length=len(featurizer)
    featurizer_dtype=featurizer.dtype
    range_start= np.arange(0,lenmols+n_split,n_split)


    range_start[-1]=lenmols

    files = [os.path.join(data_path,
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
        ignore_file = files[indices[current_start_index]][:-4]+"_ignored_indices.npy"
        current_array=(np.zeros((
            file_sizes[indices[current_start_index]],
            featurizer_length
        ))*np.nan).astype(featurizer_dtype)

        ri_max = file_sizes[indices[current_start_index]]

        return current_file,ignore_file, current_array,ri_max,current_start_index,current_start_index+ri_max

    current_file,ignore_file, current_array,ri_max,start_index,stop_index = turnover(start_index)

    ri=0
    ignored_indices=[]
    ignored_reasons={}
    dataloader.close()
    bn=os.path.basename(os.path.dirname(data_path))+"/"+os.path.basename(data_path)
    for i, mol in tqdm(enumerate(dataloader), desc=f"featzurize mols ({bn})", total=lenmols):
        if i<start_index:
            continue
        try:
            current_array[ri]=featurizer(mol,ignored_postfeaturizers=["norm"])
        except (ValueError, RuntimeError) as e:
            se=str(e)
            if se not in ignored_reasons:
                ignored_reasons[se]=[]
            ignored_reasons[se].append(ri)
            ignored_indices.append(ri)
        ri+=1
        if ri>=ri_max:
            ri=0
            np.save(ignore_file, np.array(ignored_indices,dtype=np.uint32))
            np.save(current_file, current_array)
            with open(ignore_file[:-4]+"_reasons","w+") as f:
                json.dump(ignored_reasons,f)
            ignored_indices=[]
            ignored_reasons={}
            if i<lenmols-1:
                while os.path.exists(current_file) and stop_index<lenmols:
                    current_file,ignore_file, current_array,ri_max,start_index,stop_index = turnover(stop_index)

        #print(i,indices[i],files[indices[i]])
    return True

def atom_fixed_size_featuizer_handler(featurizer:FixedSizeAtomFeaturizer,dataloader:MolDataLoader,data_path,n_split=10_000):
    lenmols=dataloader.expected_data_size
    featurizer_length=len(featurizer)
    featurizer_dtype=featurizer.dtype

    range_start= np.arange(0,lenmols+n_split,n_split)
    range_start[-1]=lenmols

    files = [os.path.join(data_path,
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
        ignore_file = files[indices[current_start_index]][:-4]+"_ignored_indices.npy"
        ri_max = file_sizes[indices[current_start_index]]

        return current_file,ignore_file,ri_max,current_start_index,current_start_index+ri_max

    current_file,ignore_file,ri_max,start_index,stop_index = turnover(start_index)

    tempmols=[]
    ri=0
    ignored_indices=[]
    ignored_reasons={}
    dataloader.close()
    for i, mol in tqdm(enumerate(dataloader), desc="featzurize atoms of mols", total=lenmols,position=1,leave=True):
        if i<start_index:
            continue
        tempmols.append(mol)

        ri+=1
        if ri>=ri_max:
            ri=0
            n_atoms=sum([m.GetNumAtoms() for m in tempmols])
            current_array=(np.zeros((
                n_atoms,
                featurizer_length
            ))*np.nan).astype(featurizer_dtype)
            atom_start_indices=np.zeros(len(tempmols),dtype=np.uint32)
            d=0
            for j,smol in enumerate(tempmols):
                atom_start_indices[j]=d
                for atom in smol.GetAtoms():
                    try:
                        current_array[d]=featurizer(atom)
                    except (ValueError, RuntimeError) as e:
                        se=str(e)
                        if se not in ignored_reasons:
                            ignored_reasons[se]=[]
                        ignored_reasons[se].append(d)
                        ignored_indices.append(d)
                    d+=1
            tempmols=[]
            np.save(ignore_file, np.array(ignored_indices,dtype=np.uint32))
            with open(ignore_file[:-4]+"_reasons","w+") as f:
                json.dump(ignored_reasons,f)
            ignored_indices=[]
            ignored_reasons={}
            np.save(current_file, current_array)
            np.save(current_file[:-4]+"_atom_start_indices.npy", atom_start_indices)
            if i<lenmols-1:
                while os.path.exists(current_file) and stop_index<lenmols:
                    current_file,ignore_file,ri_max,start_index,stop_index = turnover(stop_index)

        #print(i,indices[i],files[indices[i]])
    return True

def atom_fixed_size_iterator(data_path,featurizer):
    files = []
    for f in os.listdir(data_path):
        if not f.startswith("feats_"):
            continue
        try:
            int(f[-5:-4])
        except ValueError:
            continue

        data_fp=os.path.join(data_path,f)
        ignored_fp=os.path.join(data_path,f[:-4]+"_ignored_indices.npy")
        indices_fp=os.path.join(data_path,f[:-4]+"_atom_start_indices.npy")
        if not os.path.exists(ignored_fp):
            continue

        dd = {
            "file": data_fp,
            "start": int(f[:-4].split("_")[1]),
            "end": int(f[:-4].split("_")[2]),
            "ignored_file": ignored_fp,
            "indices_file": indices_fp,
        }
        dd["length"]=dd["end"]-dd["start"]
        files.append(dd)

    files = sorted(files,key=lambda k: k["start"])
    for f in files:
        data_array=np.load(f["file"])
        ignored_array=np.load(f["ignored_file"])
        index_array=np.load(f["indices_file"]).astype(np.uint32)
        selector = np.ones(data_array.shape[0],dtype=bool)
        selector[ignored_array]=False
        data_array=data_array[selector]

        for atom_data_per_mol in np.split(data_array,index_array[1:],axis=0):
            yield featurizer.postfeaturize(atom_data_per_mol)


def molecule_fixed_size_iterator(data_path,featurizer):
    files = []
    for f in os.listdir(data_path):
        if not f.startswith("feats_"):
            continue
        try:
            int(f[-5:-4])
        except ValueError:
            continue

        data_fp=os.path.join(data_path,f)
        ignored_fp=os.path.join(data_path,f[:-4]+"_ignored_indices.npy")
        if not os.path.exists(ignored_fp):
            continue

        dd = {
            "file": data_fp,
            "start": int(f[:-4].split("_")[1]),
            "end": int(f[:-4].split("_")[2]),
            "ignored_file": ignored_fp,
        }
        dd["length"]=dd["end"]-dd["start"]
        files.append(dd)

    files = sorted(files,key=lambda k: k["start"])

    for f in files:
        data_array=np.load(f["file"])
        ignored_array=np.load(f["ignored_file"])
        selector = np.ones(data_array.shape[0],dtype=bool)
        selector[ignored_array]=False
        data_array=data_array[selector]
        for i in range(data_array.shape[0]):
            yield featurizer.postfeaturize(data_array[i])


class UnknownFeaturizerError(NotImplementedError):
    pass

class Prefeaturizer():
    def __init__(self,dataset_name,dataloader:DataLoader,featurizer:Featurizer,featurizer_name=None,
                 featurizer_handler=None,iterator=None,
                 basedir=os.path.join(get_user_folder(), "autodata", "feats_raw_filebased")
                 ):
        self.basedir=basedir
        self.dataset_name=''.join(c for c in dataset_name if c in valid_chars)
        if featurizer_name is None:
            featurizer_name=str(featurizer)
        self.featurizer_name=''.join(c for c in featurizer_name if c in valid_chars)
        self.dataloader=dataloader

        if featurizer_handler is None:
            if not isinstance(featurizer,FixedSizeFeaturizer):
                raise UnknownFeaturizerError("featurizer must be a FixedSizeFeaturizer")

            if isinstance(featurizer,_AtomFeaturizer):
                featurizer_handler=atom_fixed_size_featuizer_handler
            elif isinstance(featurizer,_MoleculeFeaturizer):
                featurizer_handler=molecule_fixed_size_featuizer_handler
            else:
                raise UnknownFeaturizerError(f"dont know how to handle {featurizer}")

        if iterator is None:
            if not isinstance(featurizer,FixedSizeFeaturizer):
                raise UnknownFeaturizerError("featurizer must be a FixedSizeFeaturizer")
            if isinstance(featurizer,_AtomFeaturizer):
                iterator=atom_fixed_size_iterator
            elif isinstance(featurizer,_MoleculeFeaturizer):
                iterator=molecule_fixed_size_iterator
            else:
                raise UnknownFeaturizerError(f"dont know how to iterate{featurizer}")

        self.iterator=iterator
        self.featurizer_handler=featurizer_handler
        self.featurizer = featurizer

    def __len__(self):
        return self.dataloader.expected_data_size

    def _gen_path(self):
        self._data_path=os.path.join(self.basedir,self.dataset_name,self.featurizer_name)
        os.makedirs(self._data_path,exist_ok=True)
        self._info_file = os.path.join(self._data_path,"info.json")
        if not os.path.exists(self._info_file):
            with open(self._info_file,"w+") as f:
                json.dump({},f,indent=4)

    def _assert_data_path(self):
        if not hasattr(self,"_data_path"):
            self._gen_path()
        else:
            if not os.path.exists(self._data_path):
                self._gen_path()
        if not hasattr(self,"_info_file"):
            self._gen_path()
        else:
            if not os.path.exists(self._info_file):
                self._gen_path()

    @property
    def info_file(self):
        self._assert_data_path()
        return self._info_file

    @property
    def info(self):
        if not hasattr(self,"_info") or self._info is None:
            with open(self.info_file,"r") as f:
                self._info = json.load(f)
        return self._info

    @property
    def data_path(self):
        self._assert_data_path()
        return self._data_path

    def _set_info(self,key,value):
        self._info[key]=value
        with open(self._info_file,"w+") as f:
            json.dump(self._info,f,indent=4)

    def is_working(self):
        if "working" not in self.info:
            self._set_info("working",False)
        return self.info["working"]

    def is_locked(self):
        lock_file=os.path.join(self.data_path,".lock")
        return os.path.exists(lock_file)

    def lock(self):
        lock_file=os.path.join(self.data_path,".lock")
        with open(lock_file,"w+") as f:
            pass

    def unlock(self):
        lock_file=os.path.join(self.data_path,".lock")
        if os.path.exists(lock_file):
            os.remove(lock_file)

    def is_done(self):
        if "done" not in self.info:
            self._set_info("done",False)
        return self.info["done"]

    def prefeaturize(self,recalculate=False,ignore_working=False):

        self.dataloader.close() # just in case it was already in iteration


        # check and create info_file


        if self.is_working() and not ignore_working:
            MOLNET_LOGGER.error(f"prefeaturizer '{self.dataset_name}.{self.featurizer_name}' already working (info)")
            return False

        if self.is_done() and not recalculate:
            MOLNET_LOGGER.info(f"prefeaturizer '{self.dataset_name}.{self.featurizer_name}' already done")
            return True

        if self.is_locked() and not ignore_working:
            MOLNET_LOGGER.error(f"prefeaturizer '{self.dataset_name}.{self.featurizer_name}' already working (lockfile)")
            return False

        if recalculate:
            shutil.rmtree(self.data_path,ignore_errors=True)
            self._assert_data_path()
        self._set_info("done",False)
        self._set_info("working",True)

        self.lock()


        done=False
        def finish_up(done):
            self._set_info("working",False)
            self._set_info("done",done)

            self.unlock()

            return done

        try:
            done = self.featurizer_handler(featurizer=self.featurizer,dataloader=self.dataloader,data_path=self.data_path)
        except Exception as e:
            finish_up(done)
            raise e
        finish_up(done)
        return done

    def __iter__(self):
        data_path=os.path.join(self.basedir,self.dataset_name,self.featurizer_name)
        return self.iterator(data_path,self.featurizer)
