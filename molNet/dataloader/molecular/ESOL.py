import os
from tempfile import gettempdir

from tqdm import tqdm
import pandas as pd
try:
    import molNet
except ModuleNotFoundError:
    import sys,os
    modp = os.path.dirname(os.path.abspath(__file__))
    while not "molNet" in os.listdir(modp):
        modp=os.path.dirname(modp)
        if os.path.dirname(modp) == modp:
            raise ValueError("connot determine local molNet")
    if modp not in sys.path:
        sys.path.insert(0,modp)
        sys.path.append(modp)
    import molNet
    
from molNet.dataloader.molecular.dataloader import MolDataLoader
from molNet.dataloader.molecular.streamer import CSVStreamer


class ESOL(MolDataLoader):
    source = (
        "https://pubs.acs.org/doi/suppl/10.1021/ci034243x/suppl_file/ci034243xsi20040112_053635.txt"
    )
    raw_file = "delaney_data.txt"
   # expected_data_size = 440055
    data_streamer_generator = SDFStreamer.generator(
        gz=True, file_getter=lambda self: self.dataloader.raw_file_path, cached=False
    )
    
    def process_download_data(self,raw_file):
        df = pd.read_csv(raw_file)
        print(df)
        return None
    

def main():
    tdir = os.path.join(gettempdir(), "molNet", "ESOL")
    loader = ESOL(tdir,
                        data_streamer_kwargs=dict()
                        )
    loader.download()
    print(loader.expected_data_size)
    for i,m in  enumerate(tqdm(
        loader, unit="mol", unit_scale=True, total=loader.expected_data_size
    )):
        if i>100_000:
            break
    print(loader.expected_data_size)
    loader = ChemBLdb01(tdir,
                        data_streamer_kwargs=dict(addHs=True,assert_conformers=True)
                        )
    print(loader.expected_data_size)
    for i,m in  enumerate(tqdm(
            loader, unit="mol", unit_scale=True, total=loader.expected_data_size
    )):
        if i>10_000:
            break

    # loader.get_data()
    k = 1000
    loader.data_streamer.cached = True
    print(len(loader.get_n_entries(k, progress_bar=True)), flush=True)
    print(len(loader.get_n_entries(k, progress_bar=True)), flush=True)
    print(len(loader.get_n_entries(k - 1, progress_bar=True)), flush=True)
    print(len(loader.get_n_entries(k - 10, progress_bar=True)), flush=True)
    print(len(loader.get_n_entries(k - 100, progress_bar=True)), flush=True)
    print(len(loader.get_n_entries(k + 1, progress_bar=True)), flush=True)
    print(len(loader.get_n_entries(k + 10, progress_bar=True)), flush=True)

    print(len([None for mol in tqdm(
        loader, unit="mol", unit_scale=True, total=loader.expected_data_size
    )]))
    # print(mol)
    # break
    # loader.downlaod()
    # print(loader.parent_dir)


if __name__ == "__main__":
    main()
