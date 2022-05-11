from typing import List

from rdkit import Chem
from rdkit.Chem.PropertyMol import PropertyMol

from molNet.dataloader.dataloader import DataLoader
from molNet.utils.mol.properties import parallel_asset_conformers

class MolDataLoader(DataLoader):
    mol_properties:List[str]=None
    expected_mol_count:int = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.expected_mol_count is None:
            self.expected_mol_count = self.expected_data_size

    def __len__(self):
        return self.expected_mol_count

    @classmethod
    def df_smiles_to_mol(cls,df,smiles='smiles'):
        df["mol"] = df[smiles].apply(lambda s: Chem.MolFromSmiles(s))
        df.drop(df[df["mol"].apply(lambda x: x is None)].index, inplace=True)
        df["mol"] = parallel_asset_conformers(df["mol"])
        df.drop(df[df["mol"].apply(lambda x: x is None)].index, inplace=True)
        df["mol"] = df["mol"].apply(lambda m: PropertyMol(m))
        return df