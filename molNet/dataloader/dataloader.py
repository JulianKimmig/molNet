import os
import re
from typing import Callable

import requests
from tqdm import tqdm

from molNet.dataloader.streamer import DataStreamer


class DataLoader:
    source: str = None
    raw_file: str = "filename"
    expected_data_size: int = -1
    data_streamer_generator: Callable = None

    def __init__(self, parent_dir, data_streamer_kwargs=None):
        if data_streamer_kwargs is None:
            data_streamer_kwargs = {}
        os.makedirs(parent_dir, exist_ok=True)
        self._parent_dir = parent_dir
        self._data_streamer = self.data_streamer_generator(**data_streamer_kwargs)

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
        return fname

    def download(self):
        print(f"downlaod data from {self.source}")
        dl = self._downlaod()
        unpacked = self.unpack(dl)
        if unpacked != dl:
            try:
                os.remove(dl)
            except FileNotFoundError:
                pass

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
        return self._data_streamer.get_n_entries(n=n, **kwargs)

    def get_all_entries(self, **kwargs):
        self._needs_raw()
        return self._data_streamer.get_all_entries(**kwargs)

    def unpack(self, dl):
        return dl
