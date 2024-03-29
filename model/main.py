from torch import nn, optim
from torch.utils.data import DataLoader, random_split

from model.dataset import PlantDataset, getTransforms
from model.model import TraitDetector
from model.train import R2_pred, train, val_eval
from model.utils import set_device

DEVICE = set_device()


def main():
    r2_est = []
    val_loss = []
    train_loss = []
    N_EPOCHS = 30
    BATCH_SIZE = 32

    train_tf, val_tf = getTransforms()

    dataset = PlantDataset(
        "data/planttraits2024/train.csv",
        "data/planttraits2024/train_images",
        applied_transforms=train_tf,
        labeled=True,
    )

    train_dataset, val_dataset = random_split(dataset, [0.75, 0.25])
    # val_dataset = PlantDataset("data/planttraits2024/test.csv", "data/planttraits2024/test_images", applied_transforms=val_tf,
    #                            labeled=True)

    detector = TraitDetector(n_classes=6, train_features=dataset.train_columns.shape[0])

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(detector.parameters(), lr=0.0002, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
    )

    device = DEVICE  # our device (single GPU core)
    model = detector.to(device)  # put model onto the GPU core

    num_train_steps = int(len(train_dataset) / BATCH_SIZE)

    for epoch in range(N_EPOCHS):
        t_loss = train(train_loader, model, loss_fn, optimizer, device)
        y_true, y_pred, v_loss = val_eval(val_loader, model, loss_fn, device)

        r2 = R2_pred(y_pred, y_true)

        print(
            "R2: {:2.3f}, train loss: {:2.4f}, val_loss: {:2.4f}".format(
                r2, t_loss, v_loss
            )
        )
        r2_est.append(r2)

        val_loss.append(v_loss)
        train_loss.append(t_loss)

    return r2_est, val_loss, train_loss, model.cpu()
