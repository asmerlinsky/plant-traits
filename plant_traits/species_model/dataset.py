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

from plant_traits.augmentation import getTransforms
from plant_traits.constants import ID, IMG_SIZE, SD, SPECIES, TARGETS

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


class PlantSpeciesDataset(Dataset):
    """
    Dataset class for the species model. I included an augmentation method to generate and store augmented images in data and read them from there
    """

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
        full_dataset_pct_subset=None,
    ):
        """
        Reading the features dataframe to map images to species.
        Images are read into memory to speed up reading (from what i've tested it doesn't make too much of a difference though
        it's possible to get a balanced subset of the dataset

        It's also possible to train a subset of species by adapting the preprocessing notebook, but model weights won't be compatible ofc

        :param path_to_csv:
        :param path_to_imgs:
        :param path_to_species_csv:
        :param applied_transforms:
        :param labeled:
        :param full_dataset_pct_subset:
        """
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

        # It's possible to transform during training, but I'm running an augmented image dataset to run the transforms only once
        if applied_transforms:
            self.image_transforms = applied_transforms
        else:
            self.image_transforms = transforms.Compose([transforms.ToTensor()])

        self.num_species = self.species_df.shape[0]
        self.labeled = labeled

        ## Due to the dataset size, I added the option to retrieve a balanced subset
        if full_dataset_pct_subset is not None:
            self.images_df = (
                self.images_df.merge(
                    self.features_df[SPECIES], left_on=ID, right_index=True, how="left"
                )
                .groupby(SPECIES)
                .apply(lambda x: x.sample(frac=full_dataset_pct_subset))
            )
            self.images_df = self.images_df.droplevel(0).drop(axis=1, labels=[SPECIES])

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
    """
    If called, runs the augmentation, might be better to move to another module.
    """
    augment = False
    aug_tf, val_tf = getTransforms()

    if augment:

        dataset = PlantSpeciesDataset(
            "data/planttraits2024/transformed_train_df_species_1.5z_targets.csv",
            "data/planttraits2024/train_images",
            "data/planttraits2024/plant_means.csv",
            applied_transforms=aug_tf,
            labeled=True,
            # num_plants=4000,
        )

        dataset.augment(24, aug_tf, n_parallel=10)

    else:

        dataset = PlantSpeciesDataset(
            "data/planttraits2024/transformed_train_df_species_1.5z_targets.csv",
            "data/planttraits2024/train_augmented_312",
            "data/planttraits2024/plant_means.csv",
            applied_transforms=val_tf,
            labeled=True,
            # num_plants=4000,
        )

        ds_80 = dataset[80]
