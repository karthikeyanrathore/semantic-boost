from __future__ import annotations

import os

import torch
import torchvision
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

__all__ = ["CIFAR10Data"]


class CIFAR10Data:
    def __init__(
        self,
        root="./data",
        batch_size=4,
        val_ratio=0.8,
        num_workers=2,
        seed=42,
    ):
        self.root = root
        self.batch_size = batch_size
        self.val_ratio = val_ratio
        self.num_workers = num_workers
        self.seed = seed
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        self.classes = (
            "plane",
            "car",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        )
        self.trainloader = None
        self.valloader = None
        self.testloader = None

    def _ensure_data_dir(self):
        os.makedirs(self.root, exist_ok=True)

    def create_loaders(self):
        self._ensure_data_dir()
        trainset = torchvision.datasets.CIFAR10(
            root=self.root,
            train=True,
            download=True,
            transform=self.transform,
        )
        self.trainloader = DataLoader(
            trainset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

        testset = torchvision.datasets.CIFAR10(
            root=self.root,
            train=False,
            download=True,
            transform=self.transform,
        )
        total_test_samples = len(testset)
        num_val_samples = int(total_test_samples * self.val_ratio)
        num_val_samples = max(1, min(total_test_samples - 1, num_val_samples))
        num_test_samples = total_test_samples - num_val_samples
        val_dataset_split, test_dataset_split = random_split(
            testset,
            [num_val_samples, num_test_samples],
            generator=torch.Generator().manual_seed(self.seed),
        )
        self.valloader = DataLoader(
            val_dataset_split,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        self.testloader = DataLoader(
            test_dataset_split,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        print(f"self.trainloader: {len(self.trainloader)}")
        print(f"self.valloader: {len(self.valloader)}")
        print(f"self.testloader: {len(self.testloader)}")
        return self.trainloader, self.valloader, self.testloader

    def loaders(self):
        if not (self.trainloader and self.valloader and self.testloader):
            raise RuntimeError("call create_loaders() before accessing the loaders")
        return self.trainloader, self.valloader, self.testloader
