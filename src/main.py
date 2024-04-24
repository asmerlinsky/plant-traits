import sys
from os import cpu_count

import numpy as np
import torch
from torch import nn, optim, save
from torch.utils.data import DataLoader, random_split

from src.constants import BATCH_SIZE, LOG_TARGETS, N_EPOCHS
from src.model.dataset import PlantDataset, StratifiedPlantDataset, getTransforms
from src.model.models import StratifiedTraitDetector, TraitDetector
from src.model.train import R2, train, val_eval
from src.preprocess_utils import get_scaler, inverse_transform
from src.utils import benchmark_dataloader, set_device

PATH = "./stored_weights/non-strat-weights.model"
DEVICE = set_device()

from logging import INFO, FileHandler, StreamHandler, basicConfig, getLogger

file_handler = FileHandler(filename="tmp.log")
stdout_handler = StreamHandler(stream=sys.stdout)
basicConfig(level=INFO, handlers=[file_handler, stdout_handler])


logger = getLogger(__name__)


def main():
    r2_est = []
    val_loss = []
    train_loss = []

    train_tf, val_tf = getTransforms()

    dataset = StratifiedPlantDataset(
        # dataset = PlantDataset(
        "data/planttraits2024/transformed_train_df_2.5z_wo_magnitude_ouliers_targets.csv",
        "data/planttraits2024/train_images",
        applied_transforms=train_tf,
        labeled=True,
        num_plants=4000,
    )

    log_mask = torch.tensor(
        [tg in log_targets for tg in dataset.targets], dtype=torch.bool, device=DEVICE
    )

    train_dataset, val_dataset = random_split(dataset, [0.75, 0.25])
    # val_dataset = PlantDataset("data/planttraits2024/test.csv", "data/planttraits2024/test_images", applied_transforms=val_tf,
    #                            labeled=True)

    # detector = TraitDetector(
    #     n_classes=len(dataset.targets), train_features=dataset.train_columns.shape[0]
    # )
    detector = StratifiedTraitDetector(
        n_classes=len(dataset.targets),
        train_features=dataset.train_columns.shape[0],
        groups_dict=dataset.groups_dict,
    )

    loss_fn = nn.MSELoss(reduction="mean")
    # optimizer = optim.Adam(detector.parameters(), lr=0.005, weight_decay=1e-2)
    optimizer = optim.AdamW(detector.parameters(), lr=0.005, weight_decay=1e-3)
    # optimizer = optim.SGD(detector.parameters(), lr=0.001, momentum=.5, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=15, threshold_mode="rel"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=cpu_count(),
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=cpu_count(),
        pin_memory=True,
    )

    benchmark_dataloader(train_loader)

    device = DEVICE  # our device (single GPU core)
    model = detector.to(device)  # put model onto the GPU core
    standard_scaler = get_scaler("data/std_scaler.bin")
    r2 = R2(
        len(val_dataset.indices),
        len(dataset.targets),
        scaler=standard_scaler,
        log_mask=log_mask,
    )

    for epoch in range(N_EPOCHS):
        t_loss = train(
            train_loader, len(train_dataset.indices), model, loss_fn, optimizer, device
        )
        v_loss = val_eval(
            val_loader, len(val_dataset.indices), model, loss_fn, device, r2
        )

        y_pred = standard_scaler.inverse_transform(y_pred)
        y_true = standard_scaler.inverse_transform(y_pred)

        r2_vector = r2(np.exp(y_pred), np.exp(y_true))
        r2_vector = r2.pred()
        logger.info(
            f"epoch {epoch}: train loss: {t_loss:6,.5f}, val_loss: {v_loss:6,.5f}, learning_rate: {optimizer.param_groups[0]['lr']:1.6f}"
        )
        logger.info(
            "".join(
                [
                    f"{target}: {r2:6,.5f} / "
                    for target, r2 in zip(dataset.targets, r2_vector)
                ]
            )[:-3]
        )

        r2_est.append(r2_vector)
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
