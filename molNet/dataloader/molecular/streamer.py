import gzip

from rdkit import Chem
from rdkit.Chem.rdmolops import AddHs

from molNet import ConformerError
from molNet.dataloader.streamer import DataStreamer
from molNet.utils.mol.properties import assert_conformers
from molNet.utils.parallelization.multiprocessing import solve_cores


class MolStreamer(DataStreamer):
    CONF_ERROR_NONE=0
    CONF_ERROR_IGNORE=1
    CONF_ERROR_RAISE=2

    def __init__(self, *args,addHs=True,assert_conformers=True,iter_None=False,on_conformer_error=CONF_ERROR_NONE,**kwargs):
        super().__init__(*args,iter_None=iter_None,**kwargs)
        self.on_conformer_error = on_conformer_error
        self.assert_conformers = assert_conformers
        self.addHs = addHs


    def update_data(self, mol):
        if self.addHs:
            mol = AddHs(mol)
        if self.assert_conformers:
            try:
                mol = assert_conformers(mol)
            except ConformerError as e:
                if self.on_conformer_error == MolStreamer.CONF_ERROR_NONE:
                    return None
                elif self.on_conformer_error == MolStreamer.CONF_ERROR_IGNORE:
                    return mol
                elif self.on_conformer_error == MolStreamer.CONF_ERROR_RAISE:
                    raise e
                else:
                    raise ValueError("unknown conf error handling")
        return mol

class SDFStreamer(MolStreamer):
    def __init__(self, dataloader, file_getter,*args, gz=True, cached=False, threads="all-1",**kwargs):
        super(SDFStreamer, self).__init__(
            dataloader,
            *args,
            **kwargs,
            cached=cached,
            progress_bar_kwargs=dict(unit="mol", unit_scale=True),
        )
        if gz:
            threads = 1
        self._threads = threads
        self._gz = gz

        self._file_getter = file_getter

    def iterate(self):
        cores = solve_cores(self._threads)
        if cores > 1:
            sdfclasd = Chem.MultithreadedSDMolSupplier
        else:
            sdfclasd = Chem.ForwardSDMolSupplier

        def _it():
            if self._gz:
                with gzip.open(self._file_getter(self), "rb") as f:
                    for mol in sdfclasd(f):
                        if self._cached:
                            self._cache_data.append(mol)
                        yield mol
            else:
                with open(self._file_getter(self), "rb") as f:
                    for mol in sdfclasd(f):
                        if self._cached:
                            self._cache_data.append(mol)
                        yield mol
            if self._cached:
                self._all_cached = True

        return _it()
