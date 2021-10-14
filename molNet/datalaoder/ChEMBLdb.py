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


class DataStreamer:
    def __init__(self,dataloader):
        self.dataloader = dataloader

    def __iter__(self):
        pass

    @classmethod
    def generator(cls,**kwargs):
        def _generator(*args,**skwargs):
            return cls(*args,**{**kwargs,**skwargs})
        return _generator

class SDFStreamer(DataStreamer):
    def __init__(self,dataloader,file_getter,gz=True,):
        super(SDFStreamer, self).__init__(dataloader)
        self._gz = gz
        self._file_getter = file_getter

    def __iter__(self):
        def _it():
            if self._gz:
                with gzip.open(self._file_getter(self),"rb") as f:
                    for mol in Chem.ForwardSDMolSupplier(f):
                        yield mol
            else:
                with open(self._file_getter(self),"rb") as f:
                    for mol in Chem.ForwardSDMolSupplier(f):
                        yield mol
        return _it()




class Datalaoder():
    source: str = None
    raw_file: str = "filename"
    expected_data_size: int = -1
    data_streamer_generator:Callable = None

    def __init__(self, parent_dir):
        os.makedirs(parent_dir, exist_ok=True)
        self._parent_dir = parent_dir
        self._data_streamer = self.data_streamer_generator()

    def _downlaod(self) -> str:
        response = requests.get(self.source, stream=True)
        total_length = response.headers.get('content-length')
        chunk_size = 4096
        if total_length:
            total_length = int(total_length)
            # chunk_size=max(chunk_size,int(total_length/999))
            # total_length= int(total_length / chunk_size) + 1
            # print(chunk_size)

        if "Content-Disposition" in response.headers.keys():
            fname = re.findall("filename=(.+)", response.headers["Content-Disposition"])[0]
        else:
            fname = self.source.split("/")[-1]

        with open(os.path.join(self.parent_dir,fname), "wb") as handle, tqdm(total=total_length,unit="byte",unit_scale=True) as pbar:
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
        return (k for k  in self._data_streamer)

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
    source = "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/chembl_29.sdf.gz"
    raw_file = "chembl_29.sdf.gz"
    expected_data_size = 2084724
    data_streamer_generator = SDFStreamer.generator(gz=True,file_getter=lambda self: self.dataloader.raw_file_path)


def main():
    tdir=os.path.join(gettempdir(), "molNet", "ChemBLdb29")
    loader = ChemBLdb29(tdir)
    # loader.get_data()
    for mol in tqdm(loader,unit="mol",unit_scale=True):
        print(mol)
        break
    # loader.downlaod()
    # print(loader.parent_dir)


if __name__ == '__main__':
    main()
