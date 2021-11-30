import os
import re
import shutil
from typing import Callable

import requests
from tqdm import tqdm

import molNet
from molNet.dataloader.streamer import DataStreamer

CITED_SOURCES = []

#class OnlineSourceDataLoader():

#class LocalSourceDataLoader():
    

class DataLoader:
    source: str = None
    raw_file: str = None
    expected_data_size: int = -1
    data_streamer_generator: Callable = None
    citation = None
    def __init__(self, parent_dir=None, data_streamer_kwargs=None):
        if data_streamer_kwargs is None:
            data_streamer_kwargs = {}
        if parent_dir is None:
            parent_dir = os.path.join(
                molNet.get_user_folder(), "dataloader", self.__class__.__name__
            )
        os.makedirs(parent_dir, exist_ok=True)
        self._parent_dir = parent_dir
        if self.data_streamer_generator is None:
            raise ValueError(
                f"no data_streamer_generator defined for {self.__class__.__name__}"
            )
        self._data_streamer = self.data_streamer_generator(**data_streamer_kwargs)
        if self.citation is not None and self.__class__ not in CITED_SOURCES:
            molNet.MOLNET_LOGGER.info(
                f"You are using a citable datasource ('{self.__class__.__name__}'), please consider citing '{self.citation}'!")
            CITED_SOURCES.append(self.__class__)

    def _downlaod(self) -> str:
        response = requests.get(self.source, stream=True)
        total_length = response.headers.get("content-length")
        chunk_size = 4096
        if total_length:
            total_length = int(total_length)
            # chunk_size=max(chunk_size,int(total_length/999))
            # total_length= int(total_length / chunk_size) + 1
            # print(chunk_size)

        if "Content-Disposition" in response.headers.keys():
            fname = re.findall(
                "filename=(.+)", response.headers["Content-Disposition"]
            )[0]
        else:
            fname = self.source.split("/")[-1]

        with open(os.path.join(self.parent_dir, fname), "wb") as handle, tqdm(
                total=total_length, unit="byte", unit_scale=True
        ) as pbar:
            for data in response.iter_content(chunk_size=chunk_size):
                handle.write(data)
                pbar.update(len(data))
        self.process_download_data(os.path.join(self.parent_dir, fname))

        shutil.move(os.path.join(self.parent_dir, fname), self.raw_file_path)

        return fname

    def process_download_data(self, raw_file):
        return None

    def download(self):
        print(f"downlaod data from {self.source}")
        dl = self._downlaod()
        unpacked = self.unpack(dl)
        if unpacked != dl:
            try:
                os.remove(dl)
            except FileNotFoundError:
                pass
    
    def delete(self):
        if not os.path.exists(self.raw_file_path):
            return
        if os.path.isdir(self.raw_file_path):
            shutil.rmtree(self.raw_file_path)
        else:
            os.remove(self.raw_file_path)
            
    def _needs_raw(self):
        if not os.path.exists(self.raw_file_path):
            self.download()

    @property
    def data_streamer(self) -> DataStreamer:
        return self._data_streamer

    @property
    def parent_dir(self) -> str:
        return self._parent_dir

    @property
    def raw_file_path(self) -> str:
        if self.raw_file is None:
            raise ValueError(f"no raw filename defined for {self.__class__.__name__}")
        return os.path.join(self.parent_dir, self.raw_file)

    def get_data(self, force_download=False):
        self._needs_raw()
        if force_download:
            self.download()
        return [d for d in self]

    def __iter__(self):
        self._needs_raw()
        return (k for k in self._data_streamer)

    def get_n_entries(self, n: int, **kwargs):
        self._needs_raw()
        if n > self.expected_data_size:
            molNet.MOLNET_LOGGER.warning(
                f"try to get {n} entriers, but the loader has an expected size of {self.expected_data_size}")
        return self._data_streamer.get_n_entries(n=n, **kwargs)

    def get_all_entries(self, **kwargs):
        self._needs_raw()
        return self._data_streamer.get_all_entries(**kwargs)

    def unpack(self, dl):
        return dl
