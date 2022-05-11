from io import StringIO
from tempfile import gettempdir

import pandas as pd
from rdkit import Chem
from rdkit.Chem.PropertyMol import PropertyMol
from tqdm import tqdm
import os
try:
    import molNet
except ModuleNotFoundError:
    import sys

    modp = os.path.dirname(os.path.abspath(__file__))
    while not "molNet" in os.listdir(modp):
        modp = os.path.dirname(modp)
        if os.path.dirname(modp) == modp:
            raise ValueError("connot determine local molNet")
    if modp not in sys.path:
        sys.path.insert(0, modp)
        sys.path.append(modp)
    import molNet

from molNet.dataloader.molecular.dataloader import MolDataLoader
from molNet.dataloader.molecular.streamer import SDFStreamer
from molNet.utils.mol.properties import parallel_asset_conformers



class BradleyDoublePlusGoodMP(MolDataLoader):
    source = "https://s3-eu-west-1.amazonaws.com/pfigshare-u-files/1503991/BradleyDoublePlusGoodMeltingPointDataset.xlsx"
    raw_file = "BradleyDoublePlusGoodMP.sdf"
    expected_data_size = 3022
    citation = "http://dx.doi.org/10.6084/m9.figshare.1031637"
    data_streamer_generator = SDFStreamer.generator(
        gz=False, file_getter=lambda self: self.dataloader.raw_file_path, cached=False
    )
    mol_properties = ['mpC']

    def process_download_data(self, raw_file):
        df = pd.read_excel(raw_file)

        df = self.df_smiles_to_mol(df,"smiles")

        for r, d in df.iterrows():
            d["mol"].SetProp("_Name", d["name"])
            d["mol"].SetProp(
                "mpC", d["mpC"]
            )

        # needded string io since per default SDWriter appends $$$$ in the end and then a new line with results in an additional None entrie
        with StringIO() as f:
            with Chem.SDWriter(f) as w:
                for m in df["mol"]:
                    w.write(m)
            cont = f.getvalue()

        cont = "$$$$".join([c for c in cont.split("$$$$") if len(c) > 3])
        with open(raw_file, "w+") as f:
            f.write(cont)
        return raw_file


def main():
    tdir = os.path.join(gettempdir(), "molNet", "BradleyDoublePlusGoodMP")
    loader = BradleyDoublePlusGoodMP(tdir, data_streamer_kwargs=dict())

    print(loader.expected_data_size)
    for i, m in enumerate(
            tqdm(loader, unit="mol", unit_scale=True, total=loader.expected_data_size)
    ):
        pass
    print(loader.expected_data_size)


if __name__ == "__main__":
    main()
