import numpy as np
import torch

from src.constants import NUM_PASS
from src.model.models import TraitDetector


def R2_pred(y_pred, y_true):
    SS_residuals = torch.pow(y_pred - y_true, 2).sum(axis=0)
    SS_tot = torch.pow(y_true - y_true.mean(axis=0), 2).sum(axis=0)
    return 1 - SS_residuals / SS_tot


def rmse(y_pred, y_true):
    return torch.pow(y_pred - y_true, 2).sum()


def train(dataloader, model: TraitDetector, loss_fn, optimizer, device):
    model.train()
    train_loss = []
    optimizer.zero_grad()

    t_loss = 0
    for i, data in enumerate(dataloader):
        x_image, x_train, y_train = data

        x_image = x_image.to(device, dtype=torch.float)
        if isinstance(x_train, dict):
            x_train = {key: item.to(device, dtype=torch.float) for key, item in x_train.items()}
        else:
            x_train = x_train.to(device, dtype=torch.float)

        y_train = y_train.to(device, dtype=torch.float)

        train_pred = model(x_image, x_train)

        t_loss = loss_fn(train_pred, y_train)
        t_loss.backward()

        train_loss.append(t_loss.cpu().detach().numpy())

        if i % NUM_PASS == 0:

            optimizer.step()
            optimizer.zero_grad()

    model.eval()
    return np.sqrt(np.mean(train_loss))


def val_eval(dataloader, model, loss_fn, device):
    y_val_list = []
    pred_list = []
    val_loss_list = []
    for i, data in enumerate(dataloader):
        x_img, x_val, y_val = data

        x_img = x_img.to(device, dtype=torch.float)
        if isinstance(x_val, dict):
            x_val = {key: val.to(device, dtype=torch.float) for key, val in x_val.items()}
        else:
            x_val = x_val.to(device, dtype=torch.float)
        y_val = y_val.to(device, dtype=torch.float)

        pred = model(x_img, x_val)

        val_loss = loss_fn(pred, y_val)

        y_val_np = y_val.cpu().detach().numpy().tolist()
        pred_np = pred.cpu().detach().numpy().tolist()

        val_loss = val_loss.cpu().detach().numpy()

        val_loss_list.append(val_loss)
        y_val_list.extend(y_val_np)
        pred_list.extend(pred_np)
    #         print(val_loss_list)
    return (
        torch.FloatTensor(y_val_list),
        torch.FloatTensor(pred_list),
        np.sqrt(np.mean(val_loss_list)),
    )
