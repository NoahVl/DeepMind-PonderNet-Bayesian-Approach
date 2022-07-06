import os
from itertools import chain
from typing import Callable, Optional

import torch
from pl_bolts.datamodules.fashion_mnist_datamodule import FashionMNISTDataModule
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.datasets.utils import download_and_extract_archive

from datasets.parity.parity_datamodule import ParityDatamodule


class TinyImageNet200(datasets.ImageFolder):
    base_folder = "tiny-imagenet-200"
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    filename = "tiny-imagenet-200.zip"
    tgz_md5 = "90528d7ca1a48142e341f4ef8d21d0de"

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        self.train = train  # training set or test set
        self.root = root

        if download:
            self.download()

        super().__init__(
            self.datadir, transform=transform, target_transform=target_transform
        )

    @property
    def datadir(self):
        subdir = "train" if self.train else "val"
        return os.path.join(self.root, self.base_folder, subdir)

    def download(self) -> None:
        """Download the data if it doesn't exist already."""
        if os.path.exists(os.path.join(self.root, self.base_folder)):
            return

        download_and_extract_archive(
            self.url, self.root, filename=self.filename, md5=self.tgz_md5
        )

        val_dir = os.path.join(self.root, self.base_folder, "val")
        val_img_dir = os.path.join(val_dir, "images")
        val_annot_f = os.path.join(val_dir, "val_annotations.txt")

        # Open and read val annotations text file
        val_img_dict = {}
        with open(val_annot_f) as f:
            for line in f:
                words = line.split("\t")
                val_img_dict[words[0]] = words[1]

        for img, folder in val_img_dict.items():
            newpath = os.path.join(val_dir, folder)
            if not os.path.exists(newpath):
                os.makedirs(newpath)
            if os.path.exists(os.path.join(val_img_dir, img)):
                os.rename(os.path.join(val_img_dir, img), os.path.join(newpath, img))
        os.rmdir(val_img_dir)


class TinyImageNet200DataModule(LightningDataModule):
    dims = (3, 64, 64)
    num_classes = 200

    dataset_class = TinyImageNet200
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])

    def __init__(
        self,
        data_dir: str = "data",
        num_workers: int = os.cpu_count(),
        batch_size: int = 128,
        pin_memory: bool = True,
        val_split: float = 0.1,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.val_split = val_split

    def prepare_data(self):
        TinyImageNet200(self.data_dir, train=True, download=True)
        TinyImageNet200(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        data_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]
        )
        original_train_set = TinyImageNet200(
            self.data_dir,
            train=True,
            transform=data_transform,
        )

        # Deterministic stratified split.
        samples_per_class = int(len(original_train_set) / self.num_classes)
        train_split = 1 - self.val_split
        train_chunk_size = int(samples_per_class * train_split)
        train_chunk_starts = range(0, len(original_train_set), samples_per_class)
        val_chunk_starts = range(
            train_chunk_size, len(original_train_set), samples_per_class
        )
        train_indices = list(
            chain(
                *[
                    range(start, stop)
                    for start, stop in zip(train_chunk_starts, val_chunk_starts)
                ]
            )
        )
        val_indices = list(
            chain(
                *[
                    range(start, stop)
                    for start, stop in zip(
                        val_chunk_starts,
                        list(train_chunk_starts[1:]) + [len(original_train_set)],
                    )
                ]
            )
        )

        self.train = torch.utils.data.Subset(original_train_set, train_indices)
        self.val = torch.utils.data.Subset(original_train_set, val_indices)
        self.test = TinyImageNet200(
            self.data_dir,
            train=False,
            transform=data_transform,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=True,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
        )
