import gzip

from rdkit import Chem
from rdkit.Chem.rdmolops import AddHs

from molNet import ConformerError
from molNet.dataloader.streamer import DataStreamer
from molNet.utils.mol.properties import assert_conformers


class MolStreamer(DataStreamer):
    CONF_ERROR_NONE = 0
    CONF_ERROR_IGNORE = 1
    CONF_ERROR_RAISE = 2

    def __init__(
            self,
            *args,
            addHs=True,
            assert_conformers=True,
            iter_None=False,
            on_conformer_error=CONF_ERROR_NONE,
            **kwargs
    ):
        super().__init__(*args, iter_None=iter_None, **kwargs)
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
    def __init__(
            self,
            dataloader,
            file_getter,
            *args,
            gz=True,
            cached=False,
            # threads="all-1", #Not implemented due to error with closing MultithreadedSDMolSupplier
            **kwargs
    ):
        super(SDFStreamer, self).__init__(
            dataloader,
            *args,
            **kwargs,
            cached=cached,
            progress_bar_kwargs=dict(unit="mol", unit_scale=True),
        )
        if gz:
            threads = 1
        # self._threads = threads
        self._gz = gz

        self._file_getter = file_getter

    def get_iterator(self):
        # cores = solve_cores(self._threads)
        cores = 1
        if cores > 1:
            sdfclasd = Chem.MultithreadedSDMolSupplier
            filestream = False
        else:
            sdfclasd = Chem.ForwardSDMolSupplier
            filestream = True

        def _it():
            if self._gz:
                with gzip.open(self._file_getter(self), "rb") as f:
                    for mol in sdfclasd(f):
                        yield mol
            else:
                if filestream:
                    with open(self._file_getter(self), "rb") as f:
                        for mol in sdfclasd(f):
                            yield mol
                else:
                    for mol in sdfclasd(self._file_getter(self)):
                        yield mol

        return _it()
