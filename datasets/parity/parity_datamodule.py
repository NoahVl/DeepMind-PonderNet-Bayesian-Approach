import inspect
import os

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset

from .parity import ParityDataset, save_parity_data


class ParityDatamodule(pl.LightningDataModule):
    def __init__(
        self,
        num_problems: tuple[int, int, int] = (10000, 1000, 1000),
        vector_size: int = 10,
        extrapolate: bool = False,
        path: str = "data/parity/",
        batch_size: int = 32,
        num_workers: int = 0,
        shuffle: bool = True,
        pin_memory: bool = True,
        drop_last: bool = False,
        uniform: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()

        for item in inspect.signature(ParityDatamodule).parameters:
            setattr(self, item, eval(item))

        self.num_classes = 2
        self.dims = vector_size
        self.problem_str = f"{vector_size}{'_extrapolate' if extrapolate else ''}"
        self.uniform = uniform

    def prepare_data(self):
        save_parity_data(
            vector_size=self.vector_size,
            num_problems=self.num_problems,
            path=self.path,
            extrapolate=self.extrapolate,
            uniform=self.uniform,
        )

    def setup(self, stage=None):
        self.train_dataset = ParityDataset(
            os.path.join(self.path, f"train_{self.problem_str}.pt")
        )
        self.valid_dataset = ParityDataset(
            os.path.join(self.path, f"valid_{self.problem_str}.pt")
        )
        self.test_dataset = ParityDataset(
            os.path.join(self.path, f"test_{self.problem_str}.pt")
        )

    def train_dataloader(self):
        return self._data_loader(self.train_dataset, shuffle=self.shuffle)

    def val_dataloader(self):
        return self._data_loader(self.valid_dataset)

    def test_dataloader(self):
        return self._data_loader(self.test_dataset)

    def _data_loader(self, dataset: Dataset, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )
