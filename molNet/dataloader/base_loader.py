import hashlib
import os
import pickle
from shutil import copyfile
from time import time

import pandas as pd
import numpy as np

import pytorch_lightning as pl
import torch
from torch.utils.data import random_split, DataLoader, Dataset, IterableDataset

BASE_DATA_DIR = os.path.join(os.path.expanduser("~"), ".molNet", "data")


def file_hash(file):
    BUF_SIZE = 65536
    md5 = hashlib.md5()
    with open(file, 'rb') as f:
        while True:
            data = f.read(BUF_SIZE)
            if not data:
                break
            md5.update(data)
    return md5.hexdigest()


class FileDataset(Dataset):
    def __init__(self, file_list, parent_folder=None):
        self.parent_folder = parent_folder
        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, item):
        file = self.file_list[item]
        if self.parent_folder:
            file = os.path.join(self.parent_folder, file)
        with open(file, "rb") as f:
            d = pickle.load(f)
        return d


class ObjectDataset(Dataset):
    def __init__(self, objects):
        self.objects = objects

    def __len__(self):
        return len(self.objects)

    def __getitem__(self, item):
        obj = self.objects[item]
        return obj


class GeneratorDataset(IterableDataset):
    def __init__(self, generator, length=None):
        if length is None:
            length = len(generator)
        self._length = length
        self.generator = generator

    def __len__(self):
        return self._length

    def __iter__(self):
        return self

    def data_transformer(self, data):
        return data

    def __next__(self):
        return self.data_transformer(next(self.generator))

class DirectDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, collate_fn=lambda batch: batch)


class BaseLoader(pl.LightningDataModule):
    dataloader = DataLoader

    def __init__(self, split=[0.7, 0.15, 0.15], seed=None, batch_size=32):
        super().__init__()
        self.batch_size = batch_size
        self.seed = seed
        self.split = np.concatenate([np.array(split).flatten(), np.zeros(3)])[:3]
        self.split = self.split / self.split.sum()

    def generate_full_dataset(self):
        raise NotImplementedError()

    def setup(self, stage=None):
        data = self.generate_full_dataset()
        l = len(data)
        split = (self.split * l).astype(int)
        while l > split.sum():
            split[(l - split.sum()) % len(split)] += 1

        if self.seed:
            gen = torch.Generator().manual_seed(self.seed)
        else:
            gen = torch.default_generator
        self.train_ds, self.val_ds, self.test_ds = random_split(data, split, generator=gen)

    def train_dataloader(self):
        if self.train_ds is not None:
            return self.dataloader(self.train_ds, batch_size=self.batch_size)
        return None

    def val_dataloader(self):
        if self.val_ds is not None:
            return self.dataloader(self.val_ds, batch_size=self.batch_size)
        return None

    def test_dataloader(self):
        if self.test_ds is not None:
            return self.dataloader(self.test_ds, batch_size=self.batch_size)
        return None


class GeneratorDataLoader(BaseLoader):
    def __init__(self, generator, val_generator=None, test_generator=None, generatordataset_class=GeneratorDataset,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.val_generator = val_generator
        self.test_generator = test_generator
        self.generatordataset_class = generatordataset_class
        self.generator = generator

    def setup(self, stage=None):
        self.train_ds = self.generatordataset_class(self.generator)
        self.val_ds = self.generatordataset_class(self.val_generator) if self.val_generator else None
        self.test_ds = self.generatordataset_class(self.test_generator) if self.test_generator else None


class SingleFileLoader(BaseLoader):
    def __init__(self, source_file, data_dir=BASE_DATA_DIR, reload=False, save=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save = save
        self.reload = reload
        self.source_file = os.path.abspath(source_file)
        self.data_dir = data_dir

        if self.save:
            os.makedirs(self.data_dir, exist_ok=True)
            self._data_map_entry = None
            self._folder = None

    def __repr__(self):
        return "{}_{}".format(self.__class__.__name__, str(self.source_file))

    @property
    def folder(self):
        if not self.save:
            return None
        if self._folder is None:
            self._folder = os.path.join(self.data_dir, self.data_map_entry['folder'])
        return self._folder

    @property
    def data_map_entry(self):
        if not self.save:
            return None
        if self._data_map_entry is None:
            data_map = pd.read_csv(os.path.join(self.data_dir, "data_map.csv"))
            hash = file_hash(self.source_file)
            size = os.path.getsize(self.source_file)
            self._data_map_entry = \
                data_map[
                    (data_map['hash'] == hash) & (data_map['size'] == size) & (data_map['loader'] == repr(self))].iloc[
                    0]
        return self._data_map_entry

    def prepare_data(self):
        if not self.save:
            return None
        if os.path.exists(os.path.join(self.data_dir, "data_map.csv")):
            data_map = pd.read_csv(os.path.join(self.data_dir, "data_map.csv"))
        else:
            data_map = pd.DataFrame()
        if "source_file" not in data_map.columns:
            data_map['source_file'] = None
        if "hash" not in data_map.columns:
            data_map['hash'] = None
        if "size" not in data_map.columns:
            data_map['size'] = None
        if "folder" not in data_map.columns:
            data_map['folder'] = None
        if "loader" not in data_map.columns:
            data_map['loader'] = None

        hash = file_hash(self.source_file)
        size = os.path.getsize(self.source_file)

        entry = data_map[(data_map['hash'] == hash) & (data_map['size'] == size) & (data_map['loader'] == repr(self))]
        if len(entry) == 0:
            folder_base = "{}".format(hash)
            folder = folder_base
            i = 0
            while len(data_map[data_map['folder'] == folder]) > 0:
                i += 1
                folder = "{}_{}".format(folder_base, i)

            data_map = data_map.append({
                'hash': hash,
                'size': size,
                'source_file': self.source_file,
                'folder': folder,
                'loader': repr(self)
            }, ignore_index=True)

        data_map.to_csv(os.path.join(self.data_dir, "data_map.csv"), index=False)
        entry = self.data_map_entry
        os.makedirs(os.path.join(self.data_dir, entry['folder']), exist_ok=True)


class DataFrameGenerator():
    def __init__(self, df, processing=None):
        self.processing = processing
        self._length = len(df.index)
        self.df = df
        self._iter=self._continue()

    def __len__(self):
        return self._length

    def _continue(self):
        while 1:
            for r, d in self.df.iterrows():
                if self.processing:
                    d = self.processing(d)
                yield d

    def __next__(self):
        return next(self._iter)


def df_to_generator(df, split=[0.7, 0.15, 0.15], seed=None,shuffle=True, generator_class=DataFrameGenerator,*args,**kwargs):
    split = np.concatenate([np.array(split).flatten(), np.zeros(3)])[:3]
    split = split / split.sum()
    if seed is not None:
        np.random.seed(seed)
    indices = np.arange(len(df.index), dtype=int)
    if shuffle:
        np.random.shuffle(indices)
    split_cs = (np.cumsum(split) * len(df.index)).astype(int)
    return generator_class(df.iloc[indices[0:split_cs[0]]].reset_index(drop=True),*args,**kwargs), \
           generator_class(df.iloc[indices[split_cs[0]:split_cs[1]]].reset_index(drop=True),*args,**kwargs), \
           generator_class(df.iloc[indices[split_cs[1]:split_cs[2]]].reset_index(drop=True),*args,**kwargs)
