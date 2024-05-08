import sys
import time
from os import cpu_count

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import nn, optim, save
from torch.utils.data import DataLoader, Subset, random_split

from src.constants import BATCH_SIZE, LOG_TARGETS, N_EPOCHS
from src.model.dataset import getTransforms
from src.preprocess_utils import get_scaler, inverse_transform
from src.species_model.dataset import PlantDataset
from src.species_model.models import SpeciesClassifier
from src.species_model.train import train_species, val_eval
from src.utils import benchmark_dataloader, set_device

PATH = "./stored_weights/species-weights.model"
DEVICE = set_device()

from logging import INFO, FileHandler, StreamHandler, basicConfig, getLogger

file_handler = FileHandler(filename="tmp.log")
stdout_handler = StreamHandler(stream=sys.stdout)
basicConfig(level=INFO, handlers=[file_handler, stdout_handler])


logger = getLogger(__name__)


def species_main():

    val_loss = []
    train_loss = []

    train_tf, val_tf = getTransforms()

    dataset = PlantDataset(
        "data/planttraits2024/transformed_train_df_species_1.5z_targets.csv",
        "data/planttraits2024/train_augmented_312",
        path_to_species_csv="data/planttraits2024/wo_outliers_plant_means.csv",
        applied_transforms=val_tf,
        labeled=True,
    )
    logger.info(
        "Dataset size is %s. Therer are %s species", len(dataset), dataset.num_species
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
    # val_dataset = PlantDataset("data/planttraits2024/test.csv", "data/planttraits2024/test_images", applied_transforms=val_tf,
    #                            labeled=True)

    detector = SpeciesClassifier(
        n_classes=dataset.num_species,
    )
    # model = TheModelClass(*args, **kwargs)
    # model.load_state_dict(torch.load(PATH))

    loss_fn = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(detector.parameters(), lr=0.005, weight_decay=1e-2)
    optimizer = optim.AdamW(detector.parameters(), lr=0.05, weight_decay=1e-3)
    # optimizer = optim.SGD(detector.parameters(), lr=0.001, momentum=.5, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.95, patience=2, threshold_mode="rel"
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
    # model.load_state_dict(torch.load(PATH))
    # standard_scaler = get_scaler("data/std_scaler.bin")
    scaler_df = pd.read_csv("data/std_scaler.csv", index_col="targets")
    top_pct = 0.01
    topk = int(top_pct * dataset.num_species)
    for epoch in range(N_EPOCHS):
        # logger.info(f"pre train {torch.cuda.memory_allocated()}")
        start = time.time()
        t_loss, train_pos = train_species(
            train_loader, topk, model, loss_fn, optimizer, device
        )
        # logger.info(f"post train pre val {torch.cuda.memory_allocated()}")
        train_end = time.time()
        v_loss, val_pos = val_eval(val_loader, topk, model, loss_fn, device)
        # logger.info(f"post val {torch.cuda.memory_allocated()}")
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
        if epoch % 5 == 0:
            save(model.state_dict(), PATH)

    # Load:
    #
    # model = TheModelClass(*args, **kwargs)
    # model.load_state_dict(torch.load(PATH))
    # model.eval()
    #
    return val_loss, train_loss, model.cpu()
