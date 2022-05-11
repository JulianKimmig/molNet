from io import StringIO
from tempfile import gettempdir

import pandas as pd
from rdkit import Chem
from rdkit.Chem.PropertyMol import PropertyMol
from tqdm import tqdm
import os

from molNet.dataloader.molecular.dataloader import MolDataLoader
from molNet.dataloader.molecular.streamer import SDFStreamer
from molNet.utils.mol.properties import parallel_asset_conformers


class Lipo1(MolDataLoader):
    raw_file = "lipo1.sdf"
    expected_data_size = 4200
    citation = "https://doi.org/10.6019/CHEMBL3301361"
    data_streamer_generator = SDFStreamer.generator(
        gz=False, file_getter=lambda self: self.dataloader.raw_file_path, cached=False
    )
    mol_properties = ['exp']

    local_source = os.path.join(os.path.dirname(__file__),"local","Lipophilicity.csv")

    def process_download_data(self, raw_file):

        df = pd.read_csv(raw_file)
        df = self.df_smiles_to_mol(df,"smiles")
        for r, d in df.iterrows():
            d["mol"].SetProp(
                "exp", d["exp"]
            )
            d["mol"].SetProp(
                "CMPD_CHEMBLID", d["CMPD_CHEMBLID"]
            )
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
    tdir = os.path.join(gettempdir(), "molNet", "Lipo1")
    print(tdir)
    loader = Lipo1(tdir, data_streamer_kwargs=dict())

    print(loader.expected_data_size)
    for i, m in enumerate(
            tqdm(loader, unit="mol", unit_scale=True, total=loader.expected_data_size)
    ):
        pass
    print(loader.expected_data_size)


if __name__ == "__main__":
    main()
