import numpy as np
import torch

from src.constants import NUM_PASS
from src.model.models import TraitDetector
from src.utils import BATCH_SIZE, DEVICE


class R2:

    def __init__(self, val_size, num_targets, scaler, log_mask):
        self.y_pred = torch.zeros((val_size, num_targets)).to(DEVICE)
        self.y_true = torch.zeros((val_size, num_targets)).to(DEVICE)
        self.scaler = scaler
        self.log_mask = log_mask

    def inverse_transform(self):
        scaler.inverse_transform(self.y_pred)

    def pred(self):
        SS_residuals = torch.pow(self.y_pred - self.y_true, 2).sum(axis=0)
        SS_tot = torch.pow(self.y_true - self.y_true.mean(axis=0), 2).sum(axis=0)
        return 1 - SS_residuals / SS_tot


def rmse(y_pred, y_true):
    return torch.pow(y_pred - y_true, 2).sum()


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
    return np.sqrt(np.mean(train_loss))


def val_eval(dataloader, val_size, model, loss_fn, device, r2_instance: R2):

    val_loss = torch.zeros(1, device=DEVICE)

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

        r2_instance.y_pred[i : i + BATCH_SIZE] = pred
        r2_instance.y_true[i : i + BATCH_SIZE] = y_true

    return torch.sqrt(val_loss / val_size)
