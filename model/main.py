import sys

from torch import nn, optim, save
from torch.utils.data import DataLoader, random_split

from model.dataset import PlantDataset, getTransforms
from model.model import TraitDetector
from model.train import R2_pred, rmse, train, val_eval
from model.utils import set_device
from model.constants import BATCH_SIZE, SAMPLE_SIZE, N_EPOCHS
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
        "data/planttraits2024/transformed_train_df.csv",
        "data/planttraits2024/train_images",
        applied_transforms=train_tf,
        labeled=True,
        num_plants=4000
    )

    train_dataset, val_dataset = random_split(dataset, [0.75, 0.25])
    # val_dataset = PlantDataset("data/planttraits2024/test.csv", "data/planttraits2024/test_images", applied_transforms=val_tf,
    #                            labeled=True)

    detector = TraitDetector(n_classes=6, train_features=dataset.train_columns.shape[0])

    loss_fn = nn.MSELoss(reduction="mean")
    # optimizer = optim.Adam(detector.parameters(), lr=0.005, weight_decay=1e-2)
    optimizer = optim.AdamW(detector.parameters(), lr=0.01, weight_decay=1e-2)
    # optimizer = optim.SGD(detector.parameters(), lr=0.001, momentum=.5, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=20, threshold_mode="rel"
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
    )

    device = DEVICE  # our device (single GPU core)
    model = detector.to(device)  # put model onto the GPU core

    for epoch in range(N_EPOCHS):
        t_loss = train(train_loader, model, loss_fn, optimizer, scheduler, device)
        y_true, y_pred, v_loss = val_eval(val_loader, model, loss_fn, device)

        r2 = R2_pred(y_pred, y_true)

        logger.info(
            f"R2: {r2:2.3f}, train loss: {t_loss:6,.2f}, val_loss: {v_loss:6,.2f}, learning_rate: {optimizer.param_groups[0]['lr']:1.6f}"
        )

        r2_est.append(r2)

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
