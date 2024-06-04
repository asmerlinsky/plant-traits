import time

import torch
from tqdm import tqdm

from plant_traits.constants import BATCH_SIZE


def get_variable_cols(variable, columns):
    return columns[columns.str.contains(variable)]


def benchmark_dataloader(dataloader):

    N = 32
    train_dataloader_iter = iter(dataloader)
    t_start = time.perf_counter_ns()
    for _ in tqdm(range(N)):
        try:
            next(train_dataloader_iter)
        except:
            break
    n_images_per_second = (N * BATCH_SIZE) / (time.perf_counter_ns() - t_start) * 1e9
    print(f"# Images/Second: {n_images_per_second:.0f}")


def set_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("GPU is not enabled in this notebook.")
    else:
        print("GPU is enabled in this notebook. \n")

    return device


DEVICE = set_device()
print(DEVICE)


class Scaler:
    def __init__(self, scaler_df, numpy=False):
        if not numpy:
            self.mean = torch.from_numpy(scaler_df["mean"].values).to(DEVICE)
            self.std = torch.from_numpy(scaler_df["std"].values).to(DEVICE)
        else:
            self.mean = scaler_df["mean"].values
            self.std = scaler_df["std"].values

    def transform(self, mat):
        return (mat - self.mean) / self.std

    def inverse_transform(self, mat):
        return (mat + self.std) + self.mean


class R2:

    def __init__(self, val_size, num_targets, scaler_df, log_mask):
        self.y_pred = torch.zeros(
            (val_size, num_targets),
        ).to(DEVICE)
        self.y_true = torch.zeros((val_size, num_targets)).to(DEVICE)

        self.scaler = Scaler(scaler_df)
        if log_mask is not None:
            self.log_mask = torch.from_numpy(log_mask).to(DEVICE)
        else:
            self.log_mask = None

    def __call__(self):
        return self.pred()

    def inverse_transform(self, y):
        inverted = self.scaler.inverse_transform(y)
        if self.log_mask is not None:
            inverted[:, self.log_mask] = torch.exp(inverted[:, self.log_mask])
        return inverted

    def pred(self):
        orig_y_pred = self.inverse_transform(self.y_pred)
        orig_y_true = self.inverse_transform(self.y_true)
        SS_residuals = torch.pow(orig_y_pred - orig_y_true, 2).sum(axis=0)
        SS_tot = torch.pow(orig_y_true - orig_y_true.mean(axis=0), 2).sum(axis=0)
        return 1 - SS_residuals / SS_tot


def rmse(y_pred, y_true):
    return torch.pow(y_pred - y_true, 2).sum()
