import pathlib

import numpy as np
import pandas as pd
import torch
from imageio.v3 import imread
from torch.utils.data import Dataset
from torchvision import transforms

from plant_traits.constants import ID, LOG_TARGETS, SD, TARGETS
from plant_traits.species_model.dataset import PlantSpeciesDataset
from plant_traits.utils import Scaler


class PlantDataset(Dataset):
    # targets = ["X4_mean", "X11_mean", "X18_mean", "X26_mean", "X50_mean", "X3112_mean"]
    # drop_targets = ["X4_mean", "X11_mean", "X18_mean", "X50_mean", "X3112_mean"]

    # targets = ["X4_mean", "X18_mean", "X50_mean"]
    #
    # drop_targets = ["X11_mean", "X26_mean", "X3112_mean"]

    targets = TARGETS
    drop_targets = []
    sd = SD

    def __init__(
        self,
        path_to_csv,
        path_to_imgs,
        applied_transforms=None,
        labeled=False,
        num_plants=None,
    ):

        path = pathlib.Path(path_to_imgs)

        self.df = pd.read_csv(path_to_csv, dtype={"id": str})

        self.df.set_index(keys=[ID], drop=False, inplace=True)

        self.images = self.df.loc[:, ID].apply(
            lambda idx: open(path / f"{idx}.jpeg", "rb").read()
        )

        self.df.drop(axis=1, columns=[ID], inplace=True)

        if num_plants is not None:
            self.df = self.df.sample(num_plants)

        self.train_columns = self.df.columns[
            (~self.df.columns.isin(self.targets))
            & (~self.df.columns.isin(self.sd))
            & (~self.df.columns.isin(self.drop_targets))
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

        if self.image_transforms:
            image = self.image_transforms(imread(self.images[plant_id]))

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


class StratifiedPlantDataset(PlantSpeciesDataset):

    # targets = ["X4_mean", "X11_mean", "X18_mean", "X26_mean", "X50_mean", "X3112_mean"]
    targets = TARGETS
    drop_targets = []

    # targets = TARGETS
    #
    # drop_targets = []

    sd = SD

    def __init__(
        self,
        path_to_csv,
        path_to_imgs,
        path_to_species_csv,
        device,
        scaler_df,
        transform_species=True,
        applied_transforms=None,
        labeled=False,
        full_dataset_pct_subset=None,
    ):
        super().__init__(
            path_to_csv,
            path_to_imgs,
            path_to_species_csv,
            applied_transforms,
            labeled,
            full_dataset_pct_subset,
        )
        self.groups_dict = self.get_grouped_variables()
        if transform_species:
            log_mask = np.array([tg in LOG_TARGETS for tg in self.targets], dtype=bool)
            self.species_df = self.species_df[self.targets]
            self.species_df.iloc[:, log_mask] = self.species_df.iloc[:, log_mask].apply(
                np.log
            )
            self.scaler = Scaler(scaler_df, numpy=True)
            self.species_df = self.scaler.transform(self.species_df)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):

        img_id = self.images_df.index[idx]
        plant_id = self.images_df.iloc[idx][ID]

        if self.image_transforms:
            image = self.image_transforms(imread(self.images_df.loc[img_id, "image"]))

        dataset_groups = {}
        for g, variables in self.groups_dict.items():
            dataset_groups[g] = torch.from_numpy(
                self.features_df.loc[plant_id, variables].values
            )

        if self.labeled:
            return (
                image,
                dataset_groups,
                torch.from_numpy(self.features_df.loc[plant_id, self.targets].values),
            )

        return (
            image,
            dataset_groups,
        )

    def get_grouped_variables(self):

        group_dict = {}

        groups = self.train_columns.str.split("_").str[:2].str.join("_").unique()
        for g in groups:
            group_dict[g] = self.train_columns[self.train_columns.str.contains(g)]

        return group_dict
