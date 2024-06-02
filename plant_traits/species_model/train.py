import numpy as np
import torch

from plant_traits.constants import NUM_PASS
from plant_traits.species_model.models import SpeciesClassifier
from plant_traits.utils import BATCH_SIZE, DEVICE


def train_species(
    dataloader,
    topk_acc,
    model: SpeciesClassifier,
    loss_fn,
    optimizer,
    device,
    num_pass=NUM_PASS,
):
    """
    Train function, run by accumulating gradients to get a reasonable batch size
    :param dataloader:
    :param topk_acc:
    :param model:
    :param loss_fn:
    :param optimizer:
    :param device:
    :return:
    """
    model.train()
    train_loss = torch.zeros(1, device=DEVICE)
    optimizer.zero_grad()
    true_pos = 0

    for i, data in enumerate(dataloader):

        x_image, _, y_target, _ = data

        x_image = x_image.to(device, dtype=torch.float)
        y_target = y_target.to(device, dtype=torch.long)

        train_pred = model(x_image)

        t_loss = loss_fn(train_pred, y_target)
        t_loss.backward()

        train_loss += t_loss / num_pass

        tp = train_pred.topk(topk_acc)[1].t()
        true_pos += tp.eq(y_target.view(1, -1).expand_as(tp)).any(axis=0).sum().item()

        if i % num_pass == 0:

            optimizer.step()
            optimizer.zero_grad()

    model.eval()
    return train_loss.item(), true_pos


def val_eval(dataloader, topk_acc, model, loss_fn, device, num_pass=NUM_PASS):
    true_pos = 0
    val_loss = torch.zeros(1, device=DEVICE)
    with torch.no_grad():
        for i, data in enumerate(dataloader):

            x_img, _, y_target, _ = data

            x_img = x_img.to(device, dtype=torch.float)
            y_target = y_target.to(device, dtype=torch.long)

            pred = model(x_img)
            val_loss += loss_fn(pred, y_target) / num_pass

            # true_pos += (torch.argmax(pred, dim=1) == y_target).sum().item()
            tp = pred.topk(topk_acc)[1].t()
            true_pos += (
                tp.eq(y_target.view(1, -1).expand_as(tp)).any(axis=0).sum().item()
            )

    return val_loss.item(), true_pos


def batch_predict(dataloader, model, device, batch_size=BATCH_SIZE):
    predictions = torch.zeros(len(dataloader.dataset), device=device, dtype=torch.long)
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            (
                x_img,
                _,
            ) = data
            x_img = x_img.to(device, dtype=torch.float)
            predictions[i * batch_size : (i + 1) * batch_size] = model.predict(x_img)

    return predictions
