from io import StringIO
from tempfile import gettempdir

import pandas as pd
from rdkit import Chem
from rdkit.Chem.PropertyMol import PropertyMol
from tqdm import tqdm

try:
    import molNet
except ModuleNotFoundError:
    import sys, os

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


class ESOL(MolDataLoader):
    source = "https://pubs.acs.org/doi/suppl/10.1021/ci034243x/suppl_file/ci034243xsi20040112_053635.txt"
    raw_file = "delaney_data.sdf"
    expected_data_size = 1144
    data_streamer_generator = SDFStreamer.generator(
        gz=False, file_getter=lambda self: self.dataloader.raw_file_path, cached=False
    )

    def process_download_data(self, raw_file):
        df = pd.read_csv(raw_file)
        df["mol"] = df["SMILES"].apply(lambda s: Chem.MolFromSmiles(s))
        df.drop(df[df["mol"] == None].index, inplace=True)
        df["mol"] = parallel_asset_conformers(df["mol"])
        df.drop(df[df["mol"] == None].index, inplace=True)
        df["mol"] = df["mol"].apply(lambda m: PropertyMol(m))
        for r, d in df.iterrows():
            d["mol"].SetProp("_Name", d["Compound ID"])
            d["mol"].SetProp(
                "measured_log_solubility", d["measured log(solubility:mol/L)"]
            )
            d["mol"].SetProp(
                "ESOL_predicted_log_solubility",
                d["ESOL predicted log(solubility:mol/L)"],
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
        return None


def main():
    tdir = os.path.join(gettempdir(), "molNet", "ESOL")
    loader = ESOL(tdir, data_streamer_kwargs=dict())

    print(loader.expected_data_size)
    for i, m in enumerate(
            tqdm(loader, unit="mol", unit_scale=True, total=loader.expected_data_size)
    ):
        pass
    print(loader.expected_data_size)


if __name__ == "__main__":
    main()
