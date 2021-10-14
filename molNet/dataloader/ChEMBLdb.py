import gzip
import os
import re
import shutil
from tempfile import mkdtemp, gettempdir
from typing import Callable, Any, Generator

import requests
from rdkit import Chem
from rdkit.Chem.rdchem import Mol

from tqdm import tqdm

from molNet.utils.parallelization.multiprocessing import solve_cores


class DataStreamer:
    def __init__(self, dataloader, cached=False, progress_bar_kwargs=None):
        if progress_bar_kwargs is None:
            progress_bar_kwargs = {}
        self._progress_bar_kwargs = progress_bar_kwargs
        self.dataloader = dataloader
        self._cache = cached
        self._cache_data = []
        self._all_cached = False

    def __iter__(self):
        pass

    @classmethod
    def generator(cls, **kwargs):
        def _generator(*args, **skwargs):
            return cls(*args, **{**kwargs, **skwargs})

        return _generator

    def get_n_entries(self, n: int, progress_bar=False):
        if len(self._cache_data) < n and not self._all_cached:
            if progress_bar:
                g = tqdm(enumerate(self), total=n, **self._progress_bar_kwargs)
            else:
                g = enumerate(self)

            for j, d in g:
                if j >= n:
                    break
        else:
            l = len(self._cache_data)
            if l < n:
                n = l
            if progress_bar:
                return [
                    self._cache_data[i]
                    for i in tqdm(range(n), total=n, **self._progress_bar_kwargs)
                ]
        return self._cache_data[:n]


class SDFStreamer(DataStreamer):
    def __init__(self, dataloader, file_getter, gz=True, cached=False, threads="all-1"):
        super(SDFStreamer, self).__init__(
            dataloader,
            cached=cached,
            progress_bar_kwargs=dict(unit="mol", unit_scale=True),
        )
        if gz:
            threads = 1
        self._threads = threads
        self._gz = gz

        self._file_getter = file_getter

    def __iter__(self):
        cores = solve_cores(self._threads)
        if cores > 1:
            sdfclasd = Chem.MultithreadedSDMolSupplier
        else:
            sdfclasd = Chem.ForwardSDMolSupplier

        def _it():
            if self._gz:
                with gzip.open(self._file_getter(self), "rb") as f:
                    for mol in sdfclasd(f):
                        if self._cache:
                            self._cache_data.append(mol)
                        yield mol
            else:
                with open(self._file_getter(self), "rb") as f:
                    for mol in sdfclasd(f):
                        if self._cache:
                            self._cache_data.append(mol)
                        yield mol
            if self._cache:
                self._all_cached = True

        return _it()


class Datalaoder:
    source: str = None
    raw_file: str = "filename"
    expected_data_size: int = -1
    data_streamer_generator: Callable = None

    def __init__(self, parent_dir):
        os.makedirs(parent_dir, exist_ok=True)
        self._parent_dir = parent_dir
        self._data_streamer = self.data_streamer_generator()

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

    def unpack(self, dl):
        return dl


class MolDatalaoder(Datalaoder):
    autosafe = True
    store = "sdf.gz"

    @property
    def mol_path(self) -> str:
        p = os.path.join(self.parent_dir, "mol")
        os.makedirs(p, exist_ok=True)
        return p

    def get_stored_molfiles(self):
        mp = self.mol_path
        return [os.path.join(mp, f) for f in os.listdir(mp) if f.endswith(".mol")]

    @property
    def stored_molfiles(self) -> str:
        if self._stored_molfiles is None:
            self._stored_molfiles = self.get_stored_molfiles()
        return self._stored_molfiles

    def is_saved(self):
        return len(self._stored_molfiles) == self.expected_data_size

    def stream_mol_data(self) -> Generator[Mol, None, None]:
        if self.is_saved():
            for f in self.stored_molfiles:
                with open(f, "rb") as file:
                    mol = Mol(file.read())
                yield mol
        for d in self.read_raw_data():
            if not isinstance(d, Mol):  # noqa
                raise ValueError(f"{d} is not of type Mol")
            yield d


class SDFDatalaoder(MolDatalaoder):
    autosafe = False
    store = "sdf.gz"

    def read_raw_data(self) -> Chem.SDMolSupplier:
        self._needs_raw()
        suppl = Chem.SDMolSupplier(self.raw_file_path)
        return suppl


class ChemBLdb29(Datalaoder):
    source = (
        "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/chembl_29.sdf.gz"
    )
    raw_file = "chembl_29.sdf.gz"
    expected_data_size = 2084724
    data_streamer_generator = SDFStreamer.generator(
        gz=True, file_getter=lambda self: self.dataloader.raw_file_path, cached=True
    )


def main():
    tdir = os.path.join(gettempdir(), "molNet", "ChemBLdb29")
    loader = ChemBLdb29(tdir)
    # loader.get_data()
    k = 10000
    print(len(loader.get_n_entries(k, progress_bar=True)), flush=True)
    print(len(loader.get_n_entries(k, progress_bar=True)), flush=True)
    print(len(loader.get_n_entries(k - 1, progress_bar=True)), flush=True)
    print(len(loader.get_n_entries(k - 10, progress_bar=True)), flush=True)
    print(len(loader.get_n_entries(k - 100, progress_bar=True)), flush=True)
    print(len(loader.get_n_entries(k + 1, progress_bar=True)), flush=True)
    print(len(loader.get_n_entries(k + 10, progress_bar=True)), flush=True)

    for mol in tqdm(
        loader, unit="mol", unit_scale=True, total=ChemBLdb29.expected_data_size
    ):
        pass

        # print(mol)
        # break
    # loader.downlaod()
    # print(loader.parent_dir)


if __name__ == "__main__":
    main()
