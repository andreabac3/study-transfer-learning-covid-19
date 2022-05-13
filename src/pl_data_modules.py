from typing import List, Optional, Union

from numpy.core.defchararray import replace

import pytorch_lightning as pl
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Subset
import numpy as np

# our import
from torchvision.transforms import transforms

from src.select_image_folder import SelectedImageFolder
import torch

np.random.seed(42)


class BasePLDataModule(pl.LightningDataModule):
    """
    FROM LIGHTNING DOCUMENTATION

    A DataModule standardizes the training, val, test splits, data preparation and transforms.
    The main advantage is consistent data splits, data preparation and transforms across models.

    Example::

        class MyDataModule(LightningDataModule):
            def __init__(self):
                super().__init__()
            def prepare_data(self):
                # download, split, etc...
                # only called on 1 GPU/TPU in distributed
            def setup(self):
                # make assignments here (val/train/test split)
                # called on every process in DDP
            def train_dataloader(self):
                train_split = Dataset(...)
                return DataLoader(train_split)
            def val_dataloader(self):
                val_split = Dataset(...)
                return DataLoader(val_split)
            def test_dataloader(self):
                test_split = Dataset(...)
                return DataLoader(test_split)

    A DataModule implements 5 key methods:

    * **prepare_data** (things to do on 1 GPU/TPU not on every GPU/TPU in distributed mode).
    * **setup**  (things to do on every accelerator in distributed mode).
    * **train_dataloader** the training dataloader.
    * **val_dataloader** the val dataloader(s).
    * **test_dataloader** the test dataloader(s).


    This allows you to share a full dataset without explaining how to download,
    split transform and process the data

    """

    def __init__(self, conf: DictConfig):
        super().__init__()
        self.conf = conf

        self.train_dataset: Optional[SelectedImageFolder] = None
        self.validation_dataset: Optional[SelectedImageFolder] = None
        self.test_dataset: Optional[SelectedImageFolder] = None
        self.transform_no_train = transforms.Compose(
            [
                transforms.Resize(size=256),
                transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        self.trasform_train = transforms.Compose(
            [
                transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(),
                transforms.RandomHorizontalFlip(),
                transforms.CenterCrop(size=224),  # Image net standards
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # Imagenet standards
            ]
        )

    def prepare_data(self, *args, **kwargs):
        pass

    def setup(self, stage: Optional[str] = None):
        list_class_to_use = self.conf.labels.class_to_use
        train_path: str = self.conf.data.dataset.train_path
        validation_path: str = self.conf.data.dataset.validation_path
        test_path: str = self.conf.data.dataset.test_path

        self.train_dataset: SelectedImageFolder = SelectedImageFolder(
            selected_classes=list_class_to_use,
            root=train_path,
            transform=self.trasform_train,
        )
        self.validation_dataset: SelectedImageFolder = SelectedImageFolder(
            selected_classes=list_class_to_use,
            root=validation_path,
            transform=self.transform_no_train,
        )
        self.test_dataset: SelectedImageFolder = SelectedImageFolder(
            selected_classes=list_class_to_use,
            root=test_path,
            transform=self.transform_no_train,
        )

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        def get_subset_per_classes(
            classes: np.ndarray, num_samples_classes: np.ndarray, percentage: float = 0.2
        ) -> List[int]:
            """
            scrivo in italiano fanculo, comunque ritorna per ogni classe il x percentage di indici
            guarda questo esempio
            """
            curr_idx = 0
            sub_indicies = []
            for c in classes:
                class_indicies = np.random.choice(
                    a=num_samples_classes[c], size=int(num_samples_classes[c] * percentage), replace=False
                )
                sub_indicies.extend(np.add(class_indicies, curr_idx).tolist())
                curr_idx += num_samples_classes[c]
            return sub_indicies

        # qui prendo un oggetto di questo tipo
        # nel nostro caso:
        # classes = [0, 1, 2, 3]
        # num_samples_classes = [2249,   84, 1060, 1056]
        # ORA SO CHE NON `E SCRITTA AL MASSIMO AHAH chiamo questa funzoone
        classes, num_samples_classes = np.unique(self.train_dataset.targets, return_counts=True)
        # ora, quando avevo printato self.train_dataset.targets (List[int]), ritorna una lista ed erano tutti ordinati [0,0,0,0,1,1,1,1,2,2,2,2] etcc ti torna?
        # la funzione che ho scritto in effetti funziona solo se son t
        indicies = get_subset_per_classes(classes, num_samples_classes, percentage=self.conf.data.subset_percentage)

        subset = Subset(self.train_dataset, indicies)
        return DataLoader(
            subset,  # self.train_dataset,
            batch_size=self.conf.data.batch_size.train,
            shuffle=True,
            num_workers=self.conf.data.num_workers.train,
        )

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.validation_dataset,
            batch_size=self.conf.data.batch_size.val,
            num_workers=self.conf.data.num_workers.val,
        )

    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:

        return DataLoader(
            self.test_dataset,
            batch_size=self.conf.data.batch_size.test,
            num_workers=self.conf.data.num_workers.test,
        )

    def get_num_train_classes(self) -> int:
        return len(self.conf.labels.class_to_use)
