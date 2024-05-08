import numpy as np
import torch

from src.constants import NUM_PASS
from src.species_model.models import SpeciesClassifier
from src.utils import BATCH_SIZE, DEVICE


def train_species(
    dataloader, topk_acc, model: SpeciesClassifier, loss_fn, optimizer, device
):

    model.train()
    train_loss = torch.zeros(1, device=DEVICE)
    optimizer.zero_grad()
    true_pos = 0
    t_loss = 0
    for i, data in enumerate(dataloader):

        x_image, x_train, y_target, _ = data

        x_image = x_image.to(device, dtype=torch.float)
        y_target = y_target.to(device, dtype=torch.long)

        train_pred = model(x_image)

        t_loss = loss_fn(train_pred, y_target)
        t_loss.backward()

        train_loss += t_loss

        tp = train_pred.topk(topk_acc)[1].t()
        true_pos += tp.eq(y_target.view(1, -1).expand_as(tp)).any(axis=0).sum().item()

        if i % NUM_PASS == 0:

            optimizer.step()
            optimizer.zero_grad()

    model.eval()
    return train_loss.item(), true_pos


def val_eval(dataloader, topk_acc, model, loss_fn, device):
    true_pos = 0
    val_loss = torch.zeros(1, device=DEVICE)
    with torch.no_grad():
        for i, data in enumerate(dataloader):

            x_img, x_val, y_target, _ = data

            x_img = x_img.to(device, dtype=torch.float)
            y_target = y_target.to(device, dtype=torch.long)

            pred = model(x_img)
            val_loss += loss_fn(pred, y_target)

            # true_pos += (torch.argmax(pred, dim=1) == y_target).sum().item()
            tp = pred.topk(topk_acc)[1].t()
            true_pos += (
                tp.eq(y_target.view(1, -1).expand_as(tp)).any(axis=0).sum().item()
            )

    return val_loss.item(), true_pos
