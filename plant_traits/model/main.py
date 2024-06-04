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
from plant_traits.model.dataset import PlantDataset, StratifiedPlantDataset
from plant_traits.model.models import StratifiedTraitDetector, TraitDetector
from plant_traits.model.train import train, val_eval
from plant_traits.preprocess_utils import get_scaler, inverse_transform
from plant_traits.species_model.dataset import PlantSpeciesDataset
from plant_traits.utils import R2, benchmark_dataloader, set_device

PATH = "./stored_weights/non-strat-weights.model"
DEVICE = set_device()

from logging import INFO, FileHandler, StreamHandler, basicConfig, getLogger

file_handler = FileHandler(filename="tmp.log")
stdout_handler = StreamHandler(stream=sys.stdout)
basicConfig(level=INFO, handlers=[file_handler, stdout_handler])


logger = getLogger(__name__)

SPECIES_WEIGHT_PATH = "./stored_weights/species-weights18_0.8974.model"


def main():
    r2_est = []
    val_loss = []
    train_loss = []

    scaler_df = pd.read_csv("data/std_scaler_species_1.5z.csv", index_col="targets")

    dataset = StratifiedPlantDataset(
        "data/planttraits2024/transformed_train_df_species_1.5z_targets.csv",
        "data/planttraits2024/train_augmented_312",
        path_to_species_csv="data/planttraits2024/wo_outliers_plant_means.csv",
        device=DEVICE,
        scaler_df=scaler_df,
        transform_species=True,
        applied_transforms=None,
        labeled=True,
        full_dataset_pct_subset=0.2,
    )

    # log_mask = np.array([tg in LOG_TARGETS for tg in dataset.targets], dtype=bool)
    log_mask = None

    train_indexes, test_indexes = train_test_split(
        range(dataset.images_df.shape[0]),
        stratify=dataset.features_df.species.loc[dataset.images_df["id"]].values,
        random_state=2,
    )

    train_dataset, val_dataset = Subset(dataset, train_indexes), Subset(
        dataset, test_indexes
    )

    # val_dataset = PlantDataset("data/planttraits2024/test.csv", "data/planttraits2024/test_images", applied_transforms=val_tf,
    #                            labeled=True)

    # detector = TraitDetector(
    #     n_classes=len(dataset.targets), train_features=dataset.train_columns.shape[0]
    # )
    model = StratifiedTraitDetector(
        n_classes=len(dataset.targets),
        n_species=dataset.num_species,
        train_features=dataset.train_columns.shape[0],
        groups_dict=dataset.groups_dict,
        species_weights_path=SPECIES_WEIGHT_PATH,
        species_df=dataset.species_df,
        topk=50,
    )
    # model = TheModelClass(*args, **kwargs)
    # model.load_state_dict(torch.load(PATH))

    loss_fn = nn.MSELoss(reduction="mean")

    no_decay, decay = [], []
    for name, p in model.named_parameters():
        if "topk_W" in name:
            no_decay += [p]
        else:
            decay += [p]

    # optimizer = optim.Adam(detector.parameters(), lr=0.005, weight_decay=1e-2)
    optimizer = optim.AdamW(
        [
            {"params": decay, "weight_decay": 1e-3},
            {"params": no_decay, "weight_decay": 0},
        ],
        lr=0.01,
    )
    #     # optimizer = optim.SGD(detector.parameters(), lr=0.001, momentum=.5, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=5, threshold_mode="rel"
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

    # benchmark_dataloader(train_loader)

    device = DEVICE  # our device (single GPU core)
    model = model.to(device)  # put model onto the GPU core
    # model.load_state_dict(torch.load(PATH))
    # standard_scaler = get_scaler("data/std_scaler.bin")

    r2 = R2(
        len(val_dataset.indices),
        len(dataset.targets),
        scaler_df=scaler_df,
        log_mask=log_mask,
    )

    for epoch in range(N_EPOCHS):
        # logger.info(f"pre train {torch.cuda.memory_allocated()}")
        start = time.time()
        t_loss = train(
            train_loader, len(train_dataset.indices), model, loss_fn, optimizer, device
        )
        # logger.info(f"post train pre val {torch.cuda.memory_allocated()}")
        train_end = time.time()
        v_loss = val_eval(
            val_loader, len(val_dataset.indices), model, loss_fn, device, r2
        )
        # logger.info(f"post val {torch.cuda.memory_allocated()}")
        v_time = time.time() - train_end
        t_time = train_end - start
        r2_vector = r2()

        logger.info(
            f"epoch {epoch}: train loss: {t_loss:6,.5f}, val_loss: {v_loss:6,.5f},"
            + f"learning_rate: {optimizer.param_groups[0]['lr']:1.6f}\n training took {t_time/60:2.2f}, validation {v_time/60:2.2f}"
        )
        logger.info(
            "".join(
                [
                    f"{target}: {r2:6,.5f} / "
                    for target, r2 in zip(dataset.targets, r2_vector.cpu())
                ]
            )[:-3]
        )

        r2_est.append(r2_vector.cpu().detach().numpy())
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
    return r2_est, val_loss, train_loss, model.cpu()
