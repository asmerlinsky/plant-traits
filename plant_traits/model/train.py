import numpy as np
import torch

from plant_traits.constants import NUM_PASS
from plant_traits.model.models import TraitDetector
from plant_traits.utils import BATCH_SIZE, DEVICE, R2


def train(dataloader, train_size, model: TraitDetector, loss_fn, optimizer, device):
    model.train()
    train_loss = torch.zeros(1, device=DEVICE)
    optimizer.zero_grad()

    t_loss = 0
    for i, data in enumerate(dataloader):
        x_image, x_train, y_train = data

        x_image = x_image.to(device, dtype=torch.float)
        if isinstance(x_train, dict):
            x_train = {
                key: item.to(device, dtype=torch.float) for key, item in x_train.items()
            }
        else:
            x_train = x_train.to(device, dtype=torch.float)

        y_train = y_train.to(device, dtype=torch.float)

        train_pred = model(x_image, x_train)

        t_loss = loss_fn(train_pred, y_train)
        t_loss.backward()

        train_loss += t_loss

        if i % NUM_PASS == 0:

            optimizer.step()
            optimizer.zero_grad()

    model.eval()
    return np.sqrt(train_loss.item() / train_size)


def val_eval(dataloader, val_size, model, loss_fn, device, r2_instance: R2):

    val_loss = torch.zeros(1, device=DEVICE)
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            x_img, x_val, y_true = data

            x_img = x_img.to(device, dtype=torch.float)
            if isinstance(x_val, dict):
                x_val = {
                    key: val.to(device, dtype=torch.float) for key, val in x_val.items()
                }
            else:
                x_val = x_val.to(device, dtype=torch.float)
            y_true = y_true.to(device, dtype=torch.float)

            pred = model(x_img, x_val)

            val_loss += loss_fn(pred, y_true)

            starting_idx = i * BATCH_SIZE

            r2_instance.y_pred[starting_idx : starting_idx + pred.shape[0]] = pred
            r2_instance.y_true[starting_idx : starting_idx + pred.shape[0]] = y_true

    return np.sqrt(val_loss.item() / val_size)
