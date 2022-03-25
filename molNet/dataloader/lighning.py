import numpy as np

import pytorch_lightning as pl
from torch.utils.data import Subset, DataLoader


class InMemoryLoader(pl.LightningDataModule):

    def __init__(
            self,
            data,
            split=[0.8, 0.1, 0.1],
            batch_size=32,
            dataloader=DataLoader,
            **dataloader_kwargs
    ):
        super().__init__()
        self.batch_size = batch_size
        self.split = np.concatenate([np.array(split).flatten(), np.zeros(3)])[:3]
        self.split = self.split / self.split.sum()
        self.dataloader_kwargs = dataloader_kwargs
        self.dataloader = dataloader
        self.data=data

    def setup(self, stage=None):
        data = self.data
        l = len(data)
        split = (self.split * l).astype(int)
        while l > split.sum():
            split[((l - split.sum()) % len(split))] += 1

        indices = np.arange(sum(split))
        self.train_ds, self.val_ds, self.test_ds = [
            Subset(data, indices[offset - length : offset])
            for offset, length in zip(np.add.accumulate(split), split)
        ]

    def train_dataloader(self):
        if self.train_ds is not None:
            return self.dataloader(
                self.train_ds, batch_size=self.batch_size, **self.dataloader_kwargs
            )
        return None

    def val_dataloader(self):
        if self.val_ds is not None:
            return self.dataloader(
                self.val_ds, batch_size=self.batch_size, **self.dataloader_kwargs
            )
        return None

    def test_dataloader(self):
        if self.test_ds is not None:
            return self.dataloader(
                self.test_ds, batch_size=self.batch_size, **self.dataloader_kwargs
            )
        return None