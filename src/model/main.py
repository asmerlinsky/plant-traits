import sys

from torch import nn, optim, save
from torch.utils.data import DataLoader, random_split

from src.constants import BATCH_SIZE, N_EPOCHS
from src.model.dataset import PlantDataset, getTransforms
from src.model.model import TraitDetector
from src.model.train import R2_pred, train, val_eval
from src.utils import set_device

PATH = "./stored_weights/weights.model"
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

    dataset = PlantDataset(
        "data/planttraits2024/transformed_train_df_2z.csv",
        "data/planttraits2024/train_images",
        applied_transforms=train_tf,
        labeled=True,
    )

    train_dataset, val_dataset = random_split(dataset, [0.75, 0.25])
    # val_dataset = PlantDataset("data/planttraits2024/test.csv", "data/planttraits2024/test_images", applied_transforms=val_tf,
    #                            labeled=True)

    detector = TraitDetector(
        n_classes=len(dataset.targets), train_features=dataset.train_columns.shape[0]
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
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    device = DEVICE  # our device (single GPU core)
    model = detector.to(device)  # put model onto the GPU core

    for epoch in range(N_EPOCHS):
        t_loss = train(train_loader, model, loss_fn, optimizer, device)
        y_true, y_pred, v_loss = val_eval(val_loader, model, loss_fn, device)

        r2_vector = R2_pred(y_pred, y_true)

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

    save(model.state_dict(), PATH)

    # Load:
    #
    # model = TheModelClass(*args, **kwargs)
    # model.load_state_dict(torch.load(PATH))
    # model.eval()
    #
    return r2_est, val_loss, train_loss, model.cpu()
