import json
import shutil
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


class FreeSolv_0_51(MolDataLoader):
    source = "https://escholarship.org/content/qt6sd403pz/supp/FreeSolv-0.51.zip"
    raw_file = "freesolv-0.51.sdf"
    expected_data_size = 643
    citation = "https://doi.org/10.1007/s10822-014-9747-x"
    data_streamer_generator = SDFStreamer.generator(
        gz=False, file_getter=lambda self: self.dataloader.raw_file_path, cached=False
    )
    mol_properties = ["expt","d_expt","calc","d_calc"]

        
    def process_download_data(self, raw_file):
        import zipfile
        with zipfile.ZipFile(raw_file, 'r') as zip_ref:
            zip_ref.extractall(os.path.dirname(raw_file))
        files_to_del=list(os.listdir(os.path.dirname(raw_file)))
        with open(os.path.join(os.path.dirname(raw_file), "FreeSolv-0.51","database.json"), "r") as f:
            dict= json.loads(f.read())

        for f in files_to_del:
            if os.path.isfile(os.path.join(os.path.dirname(raw_file), f)):
                os.remove(os.path.join(os.path.dirname(raw_file), f))
            elif os.path.isdir(os.path.join(os.path.dirname(raw_file), f)):
                shutil.rmtree(os.path.join(os.path.dirname(raw_file), f))

        df = pd.DataFrame.from_dict(dict, orient="index")
        df=df[["iupac","smiles", "expt","d_expt","calc","d_calc"]]

        df["mol"] = df["smiles"].apply(lambda s: Chem.MolFromSmiles(s))
        df.drop(df[df["mol"] == None].index, inplace=True)
        df["mol"] = parallel_asset_conformers(df["mol"])
        df.drop(df[df["mol"] == None].index, inplace=True)
        df["mol"] = df["mol"].apply(lambda m: PropertyMol(m))
        for r, d in df.iterrows():
            d["mol"].SetProp("_Name", d["iupac"])
            d["mol"].SetProp(
                "expt", d["expt"]
            )
            d["mol"].SetProp(
                "d_expt", d["d_expt"]
            )
            d["mol"].SetProp(
                "calc", d["calc"]
            )
            d["mol"].SetProp(
                "d_calc", d["d_calc"]
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

FreeSolv=FreeSolv_0_51
def main():
    tdir = os.path.join(gettempdir(), "molNet", "FreeSolv_0.51")
    print(tdir)
    print(tdir)
    loader = FreeSolv(tdir, data_streamer_kwargs=dict())

    print(loader.expected_data_size)
    for i, m in enumerate(
            tqdm(loader, unit="mol", unit_scale=True, total=loader.expected_data_size)
    ):
        pass
    print(loader.expected_data_size)


if __name__ == "__main__":
    main()
