import os
from tempfile import gettempdir

from tqdm import tqdm

from molNet.dataloader.molecular.dataloader import MolDataLoader
from molNet.dataloader.molecular.streamer import SDFStreamer

class Tox21Train(MolDataLoader):
    source = "https://tripod.nih.gov/tox21/challenge/download?id=tox21_10k_data_allsdf"
    raw_file = "tox21_10k_data_all.sdf"
    expected_data_size = 11764
    expected_mol_count = 11758
    data_streamer_generator = SDFStreamer.generator(
        gz=False, file_getter=lambda self: self.dataloader.raw_file_path, cached=False
    )

    mol_properties=['NR-AR',
                        'NR-AR-LBD',
                        'NR-AhR',
                        'NR-Aromatase',
                        'NR-ER',
                        'NR-ER-LBD',
                        'NR-PPAR-gamma',
                        'SR-ARE',
                        'SR-ATAD5',
                        'SR-HSE',
                        'SR-MMP',
                        'SR-p53']

    def process_download_data(self, raw_file):
        import zipfile
        with zipfile.ZipFile(raw_file,"r") as zip_ref:
            zip_ref.extractall(os.path.dirname(raw_file))
        os.remove(raw_file)
        return raw_file.rsplit(".zip", 1)[0]

Tox21=Tox21Train
def main():
    tdir = os.path.join(gettempdir(), "molNet", "Tox21")#
    print(tdir)
    loader = Tox21(tdir,data_streamer_kwargs=dict(iter_None=True))
    print(loader.expected_data_size)

    _all_mps={p:[] for p in all_props}
    for i, m in enumerate(
            tqdm(loader, unit="mol", unit_scale=True, total=loader.expected_data_size)
    ):
        if m:
            md = m.GetPropsAsDict()
            for p in all_props:
                _all_mps[p].append(md.get(p,None))
            #print(m.GetPropsAsDict().keys())
    print(_all_mps)
    print(loader.expected_data_size)

    print(
        len(
            [
                None
                for mol in tqdm(
                loader, unit="mol", unit_scale=True, total=loader.expected_data_size
            )
            ]
        )
    )
    return _all_mps


if __name__ == "__main__":
    _all_mps = main()