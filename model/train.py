import numpy as np
import torch

from model.model import TraitDetector


def R2_pred(y_pred, y_true):
    SS_residuals = torch.pow(y_pred - y_true, 2).sum()
    SS_tot = torch.pow(y_true - y_true.mean(axis=0), 2).sum()
    return 1 - SS_residuals / SS_tot


def train(dataloader, model: TraitDetector, loss_fn, optimizer, device):
    model.train()
    train_loss = []

    for i, data in enumerate(dataloader):
        x_image, x_train, y_train = data

        x_image = x_image.to(device, dtype=torch.float)
        x_train = x_train.to(device, dtype=torch.float)
        y_train = y_train.to(device, dtype=torch.float)

        optimizer.zero_grad()

        train_pred = model(x_image, x_train)

        t_loss = loss_fn(train_pred, y_train)

        t_loss.backward()

        optimizer.step()
        train_loss.append(t_loss.cpu().detach().numpy())
    #         tk0.set_postfix(t_loss=train_loss[-1])

    model.eval()
    return np.mean(train_loss)


def val_eval(dataloader, model, loss_fn, device):
    y_val_list = []
    pred_list = []
    val_loss_list = []
    for i, data in enumerate(dataloader):
        x_img, x_val, y_val = data

        x_img = x_img.to(device, dtype=torch.float)
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
        np.mean(val_loss_list),
    )
