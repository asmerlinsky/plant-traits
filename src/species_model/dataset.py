import os
import pathlib
from logging import getLogger
from time import time

import numpy as np
import pandas as pd
import torch
from imageio.v3 import imread
from joblib import Parallel, delayed
from PIL import Image
from torch.nn.functional import one_hot
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm

from src.constants import ID, IMG_SIZE, SD, SPECIES, TARGETS
from src.model.dataset import getTransforms

logger = getLogger(__name__)


def augment_func(
    i,
    species_counts,
    features_df,
    images_df,
    aug_path,
    img_per_specie,
    aug_transform,
    orig_transform,
):
    for specie, num_images in tqdm(species_counts.items(), position=i):
        ids = features_df[features_df[SPECIES] == specie].index

        for _id in ids:
            tf = orig_transform(imread(images_df.loc[_id, "image"]))

            save_image(tf, f"{aug_path}/{_id}_orig.jpeg")

        remaining_augs = img_per_specie - num_images
        num_iter = int(np.floor(remaining_augs / num_images))
        mod = remaining_augs % num_images

        for i in range(num_iter):
            for j in range(num_images):
                tf = aug_transform(imread(images_df.loc[ids[j], "image"]))
                save_image(tf, f"{aug_path}/{ids[j]}_aug{i}.jpeg")

        for id_ in np.random.choice(ids, replace=False, size=mod):
            tf = aug_transform(imread(images_df.loc[id_, "image"]))
            save_image(tf, f"{aug_path}/{id_}_aug{i}.jpeg")


class PlantDataset(Dataset):
    # targets = ["X4_mean", "X11_mean", "X18_mean", "X26_mean", "X50_mean", "X3112_mean"]
    # drop_targets = ["X4_mean", "X11_mean", "X18_mean", "X50_mean", "X3112_mean"]

    # targets = ["X4_mean", "X18_mean", "X50_mean"]
    #
    # drop_targets = ["X11_mean", "X26_mean", "X3112_mean"]

    targets = TARGETS
    drop_targets = []
    sd = SD
    species_col = SPECIES

    def __init__(
        self,
        path_to_csv,
        path_to_imgs,
        path_to_species_csv,
        applied_transforms=None,
        labeled=False,
        num_plants=None,
        img_size=None,
    ):
        logger.info("Loading dataset")
        path = pathlib.Path(path_to_imgs)

        self.species_df = pd.read_csv(
            path_to_species_csv, dtype={SPECIES: str}, index_col=SPECIES
        )

        self.features_df = pd.read_csv(path_to_csv, dtype={ID: str})
        self.features_df.set_index(keys=[ID], drop=False, inplace=True)
        self.features_df.drop(axis=1, columns=[ID], inplace=True)

        self.train_columns = self.features_df.columns[
            ~self.features_df.columns.isin(
                self.targets + self.sd + self.drop_targets + [self.species_col]
            )
        ]

        img_list = os.listdir(path)
        images_df = pd.DataFrame(img_list, columns=["image"])

        images_df.index = images_df["image"].str.split(".").str[0]
        images_df.index.name = "image_id"

        images_df[ID] = images_df.index.str.split("_").str[0]
        images_df["image"] = images_df["image"].apply(
            lambda filename: open(path / f"{filename}", "rb").read()
        )

        self.images_df = images_df

        self.images_df = self.images_df[self.images_df[ID].isin(self.features_df.index)]

        if applied_transforms:
            self.image_transforms = applied_transforms
        else:
            self.image_transforms = transforms.Compose([transforms.ToTensor()])

        self.num_species = self.species_df.shape[0]
        self.labeled = labeled
        logger.info("Done!")

    def __len__(self):
        return self.images_df.shape[0]

    def __getitem__(self, idx):

        img_id = self.images_df.index[idx]
        plant_id = self.images_df.iloc[idx][ID]

        if self.image_transforms:
            image = self.image_transforms(imread(self.images_df.loc[img_id, "image"]))

        if self.labeled:
            return (
                image,
                torch.from_numpy(
                    self.features_df.loc[plant_id, self.train_columns].values
                ),
                self.features_df.loc[plant_id, self.species_col],
                torch.from_numpy(self.features_df.loc[plant_id, self.targets].values),
            )

        return (
            image,
            torch.from_numpy(self.features_df.loc[img_id, self.train_columns].values),
            None,
            None,
        )

    def augment(
        self,
        img_per_specie,
        aug_transform,
        aug_path=f"data/planttraits2024/train_augmented_{IMG_SIZE}",
        n_parallel=3,
    ):
        start = time()
        species_counts = self.features_df[SPECIES].value_counts()
        orig_transform = transforms.Compose(
            (transforms.ToTensor(), transforms.Resize(size=IMG_SIZE))
        )

        #
        # num_groups = round(np.ceil(species_counts.shape[0]/n_parallel))
        # shuffled = species_counts.sample(frac=1)
        # species_count_groups = [shuffled.iloc[i: i+num_groups].copy() for i in range(0, species_counts.shape[0], num_groups)]
        #
        # Parallel(n_jobs=n_parallel, verbose=50, backend='threading')(delayed(augment_func)(i+1, species_counts_subset,
        #                                                                      self.features_df[self.features_df[SPECIES].isin(species_counts_subset.index)],
        #                                                                      self.images_df[self.images_df[ID].isin(self.features_df[self.features_df[SPECIES].isin(species_counts_subset.index)].index)],
        #                                                                      aug_path,
        #                                                                      img_per_specie,
        #                                                                      aug_transform,
        #                                                                      orig_transform)
        #
        #                    for i, species_counts_subset in enumerate(species_count_groups)
        #                                 )
        #
        for specie, num_images in tqdm(species_counts.items()):
            ids = self.features_df[self.features_df[SPECIES] == specie].index

            for _id in ids:
                tf = orig_transform(imread(self.images_df.loc[_id, "image"]))

                save_image(tf, f"{aug_path}/{_id}_orig.jpeg")

            remaining_augs = img_per_specie - num_images
            num_iter = int(np.floor(remaining_augs / num_images))
            mod = remaining_augs % num_images

            for i in range(num_iter):
                for j in range(num_images):
                    tf = aug_transform(imread(self.images_df.loc[ids[j], "image"]))
                    save_image(tf, f"{aug_path}/{ids[j]}_aug{i}.jpeg")

            for id_ in np.random.choice(ids, replace=False, size=mod):
                tf = aug_transform(imread(self.images_df.loc[id_, "image"]))
                save_image(tf, f"{aug_path}/{id_}_aug{i}.jpeg")

        print(time() - start)


# def save_jpeg(tensor, )

if __name__ == "__main__":
    augment = False
    aug_tf, val_tf = getTransforms()

    if augment:

        dataset = PlantDataset(
            "data/planttraits2024/transformed_train_df_species_1.5z_targets.csv",
            "data/planttraits2024/train_images",
            "data/planttraits2024/plant_means.csv",
            applied_transforms=aug_tf,
            labeled=True,
            # num_plants=4000,
        )

        dataset.augment(24, aug_tf, n_parallel=10)

    else:

        dataset = PlantDataset(
            "data/planttraits2024/transformed_train_df_species_1.5z_targets.csv",
            "data/planttraits2024/train_augmented_312",
            "data/planttraits2024/plant_means.csv",
            applied_transforms=val_tf,
            labeled=True,
            # num_plants=4000,
        )

        ds_80 = dataset[80]
