import os
from tempfile import gettempdir

from tqdm import tqdm

from molNet.dataloader.molecular.dataloader import MolDataLoader
from molNet.dataloader.molecular.streamer import SDFStreamer


class ChemBLdb01(MolDataLoader):
    source = (
        "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_01/chembl_01.sdf.gz"
    )
    raw_file = "chembl_01.sdf.gz"
    expected_data_size = 440055
    data_streamer_generator = SDFStreamer.generator(
        gz=True, file_getter=lambda self: self.dataloader.raw_file_path, cached=False
    )


class ChemBLdb29(MolDataLoader):
    source = (
        "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_29/chembl_29.sdf.gz"
    )
    raw_file = "chembl_29.sdf.gz"
    expected_data_size = 2084724
    data_streamer_generator = SDFStreamer.generator(
        gz=True, file_getter=lambda self: self.dataloader.raw_file_path, cached=False
    )


def main():
    tdir = os.path.join(gettempdir(), "molNet", "ChemBLdb01")
    loader = ChemBLdb01(tdir,
                        data_streamer_kwargs=dict()
                        )
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
