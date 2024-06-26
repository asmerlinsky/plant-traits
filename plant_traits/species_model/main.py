import sys
import time
from os import cpu_count

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import nn, optim, save
from torch.utils.data import DataLoader, Subset

from plant_traits.augmentation import getTransforms
from plant_traits.constants import BATCH_SIZE, LOG_TARGETS, N_EPOCHS
from plant_traits.species_model.dataset import PlantSpeciesDataset
from plant_traits.species_model.models import SpeciesClassifier
from plant_traits.species_model.train import batch_predict, train_species, val_eval
from plant_traits.utils import benchmark_dataloader, set_device

# READ_PATH = "./stored_weights/species-weights4.model"
READ_PATH = "./stored_weights/species-weights18_0.8974.model"
# SAVE_PATH = "./stored_weights/species-weights5.model"
DEVICE = set_device()

from logging import INFO, FileHandler, StreamHandler, basicConfig, getLogger

file_handler = FileHandler(filename="tmp.log")
stdout_handler = StreamHandler(stream=sys.stdout)
basicConfig(level=INFO, handlers=[file_handler, stdout_handler])


logger = getLogger(__name__)


def species_main():

    val_loss = []
    train_loss = []

    dataset = PlantSpeciesDataset(
        "data/planttraits2024/transformed_train_df_species_1.5z_targets.csv",
        "data/planttraits2024/train_augmented_312",
        path_to_species_csv="data/planttraits2024/wo_outliers_plant_means.csv",
        applied_transforms=None,
        labeled=True,
        # full_dataset_pct_subset=0.2
    )
    logger.info(
        "Dataset size is %s. There are %s species", len(dataset), dataset.num_species
    )

    ## Balancing dataset, since there are ~17000 classes
    train_indexes, test_indexes = train_test_split(
        range(dataset.images_df.shape[0]),
        stratify=dataset.features_df.species.loc[dataset.images_df["id"]].values,
        random_state=2,
    )
    train_dataset, val_dataset = Subset(dataset, train_indexes), Subset(
        dataset, test_indexes
    )

    detector = SpeciesClassifier(n_classes=dataset.num_species)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(detector.parameters(), lr=0.00005, weight_decay=1e-3)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, patience=2, threshold_mode="rel", threshold=1e-3
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=cpu_count() - 2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=cpu_count() - 2,
        pin_memory=True,
    )

    benchmark_dataloader(train_loader)

    device = DEVICE  # our device (single GPU core)
    model = detector.to(device)  # put model onto the GPU core
    model.load_state_dict(torch.load(READ_PATH))

    top_pct = 1 / dataset.num_species
    topk = int(top_pct * dataset.num_species)
    j = 0
    for epoch in range(N_EPOCHS):

        start = time.time()
        t_loss, train_pos = train_species(
            train_loader, topk, model, loss_fn, optimizer, device
        )

        train_end = time.time()
        v_loss, val_pos = val_eval(val_loader, topk, model, loss_fn, device)

        v_time = time.time() - train_end
        t_time = train_end - start

        logger.info(
            f"epoch {epoch}: train loss: {t_loss:6,.5f},({train_pos/len(train_dataset.indices):1.4f} top{topk} acc)"
            + f"val_loss: {v_loss:6,.5f} ({val_pos/len(val_dataset.indices):1.4f} top{topk} acc),"
            + f"learning_rate: {optimizer.param_groups[0]['lr']:1.6f}\n training took {t_time/60:2.2f}, validation {v_time/60:2.2f}"
        )

        val_loss.append(v_loss)
        train_loss.append(t_loss)
        scheduler.step(v_loss)
        if epoch % 1 == 0:
            save(
                model.state_dict(),
                f"./stored_weights/species-weights{j}_{val_pos/len(val_dataset.indices):1.4f}.model",
            )
            j += 1

    return val_loss, train_loss, model.cpu()


def test_lr():

    val_loss = []
    train_loss = []

    train_tf, val_tf = getTransforms()

    dataset = PlantSpeciesDataset(
        "data/planttraits2024/transformed_train_df_species_1.5z_targets.csv",
        "data/planttraits2024/train_augmented_312",
        path_to_species_csv="data/planttraits2024/wo_outliers_plant_means.csv",
        applied_transforms=None,
        labeled=True,
        # full_dataset_pct_subset=0.2,
    )
    logger.info(
        "Dataset size is %s. There are %s species", len(dataset), dataset.num_species
    )
    log_mask = np.array([tg in LOG_TARGETS for tg in dataset.targets], dtype=bool)
    dataset.features_df.species

    train_indexes, test_indexes = train_test_split(
        range(dataset.images_df.shape[0]),
        stratify=dataset.features_df.species.loc[dataset.images_df["id"]].values,
    )

    train_dataset, val_dataset = Subset(dataset, train_indexes), Subset(
        dataset, test_indexes
    )

    loss_fn = nn.CrossEntropyLoss()

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=cpu_count() - 2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=cpu_count() - 2,
        pin_memory=True,
    )

    benchmark_dataloader(train_loader)

    device = DEVICE  # our device (single GPU core)

    scaler_df = pd.read_csv("data/std_scaler.csv", index_col="targets")
    top_pct = 0.1
    topk = int(top_pct * dataset.num_species)

    learning_rates = np.logspace(-5, -3, 8)
    loss_dict = {}
    weight_decays = [1e-3]
    loss_list = []
    for lr in learning_rates:
        for wd in weight_decays:
            detector = SpeciesClassifier(
                n_classes=dataset.num_species,
            )
            optimizer = optim.AdamW(detector.parameters(), lr=lr, weight_decay=wd)
            model = detector.to(device)  # put model onto the GPU core
            results = {}
            for epoch in range(3):

                start = time.time()
                t_loss, train_pos = train_species(
                    train_loader, topk, model, loss_fn, optimizer, device
                )

                train_end = time.time()
                v_loss, val_pos = val_eval(val_loader, topk, model, loss_fn, device)

                v_time = time.time() - train_end
                t_time = train_end - start

                logger.info(
                    f"epoch {epoch}: train loss: {t_loss:6,.5f},({train_pos/len(train_dataset.indices):1.4f} top{topk} acc)"
                    + f"val_loss: {v_loss:6,.5f} ({val_pos/len(val_dataset.indices):1.4f} top{topk} acc),"
                    + f"learning_rate: {optimizer.param_groups[0]['lr']:1.6f}\n training took {t_time/60:2.2f}, validation {v_time/60:2.2f}"
                )

                val_loss.append(v_loss)
                train_loss.append(t_loss)
            loss_dict[lr, wd] = (train_loss, val_loss)
    return loss_dict


def predict_targets():
    predict_batch_size = 128

    test_dataset = PlantSpeciesDataset(
        "data/planttraits2024/transformed_test_df_species_1.5z_targets.csv",
        "data/planttraits2024/test_images",
        path_to_species_csv="data/planttraits2024/wo_outliers_plant_means.csv",
        applied_transforms=None,
        labeled=False,
        # full_dataset_pct_subset=0.2
    )
    logger.info(
        "Dataset size is %s. There are %s species",
        len(test_dataset),
        test_dataset.num_species,
    )

    ## Balancing dataset, since there are ~17000 classes

    detector = SpeciesClassifier(
        n_classes=test_dataset.num_species,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=predict_batch_size,
        shuffle=False,
        num_workers=cpu_count() - 2,
        pin_memory=True,
    )

    device = DEVICE  # our device (single GPU core)
    model = detector.to(device)  # put model onto the GPU core
    model.load_state_dict(torch.load(READ_PATH))

    predictions = batch_predict(test_loader, model, DEVICE, predict_batch_size)

    targets = test_dataset.species_df.loc[predictions.to("cpu").numpy().astype("str")][
        test_dataset.targets
    ]

    targets = targets.rename(
        {col: col.split("_")[0] for col in targets.columns}, axis=1
    )

    return targets
