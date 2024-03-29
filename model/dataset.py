import pathlib

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

SIZE = 512


class PlantDataset(Dataset):

    targets = ["X4_mean", "X11_mean", "X18_mean", "X26_mean", "X50_mean", "X3112_mean"]
    sd = ["X4_sd", "X11_sd", "X18_sd", "X26_sd", "X50_sd", "X3112_sd"]

    def __init__(
        self,
        path_to_csv,
        path_to_imgs,
        applied_transforms=None,
        labeled=False,
        num_plants=None,
    ):

        self.path = pathlib.Path(path_to_imgs)

        self.df = pd.read_csv(path_to_csv, dtype={"id": str})

        self.df.set_index(keys=["id"], drop=True, inplace=True)

        if num_plants is not None:
            self.df = self.df.iloc[:num_plants]

        self.train_columns = self.df.columns[
            (~self.df.columns.isin(self.targets)) & (~self.df.columns.isin(self.sd))
        ]

        if applied_transforms:
            self.image_transforms = applied_transforms
        else:
            self.image_transforms = transforms.Compose([transforms.ToTensor()])

        self.labeled = labeled

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):

        plant_id = self.df.index[idx]

        image = Image.open(self.path / f"{plant_id}.jpeg")
        if self.image_transforms:
            image = self.image_transforms(image)

        if self.labeled:
            return (
                image,
                torch.from_numpy(self.df.loc[plant_id, self.train_columns].values),
                torch.from_numpy(self.df.loc[plant_id, self.targets].values),
            )

        return (
            image,
            torch.from_numpy(self.df.loc[plant_id, self.train_columns].values),
            None,
        )


def getTransforms():

    first_transform = [transforms.ToTensor()]

    aug_transforms = [
        transforms.RandomResizedCrop(size=SIZE),
        transforms.RandomRotation(degrees=180),
    ]

    preprocessing_transforms = [  # T.ToTensor(),
        transforms.Resize(size=SIZE),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
    train_transformer = transforms.Compose(
        first_transform + aug_transforms + preprocessing_transforms
    )
    val_transformer = transforms.Compose(first_transform + preprocessing_transforms)
    return train_transformer, val_transformer
