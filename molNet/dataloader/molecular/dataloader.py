from typing import List

from molNet.dataloader.dataloader import DataLoader


class MolDataLoader(DataLoader):
    mol_properties:List[str]=None
    expected_mol_count:int = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.expected_mol_count is None:
            self.expected_mol_count = self.expected_data_size

    def __len__(self):
        return self.expected_mol_count