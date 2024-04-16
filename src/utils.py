import torch
import time
from src.constants import BATCH_SIZE
from tqdm import tqdm

def get_variable_cols(variable, columns):
    return columns[columns.str.contains(variable)]

def benchmark_dataloader(dataloader):

    N = 32
    train_dataloader_iter = iter(dataloader)
    t_start = time.perf_counter_ns()
    for _ in tqdm(range(N)):
        next(train_dataloader_iter)
    n_images_per_second = (N * BATCH_SIZE) / (time.perf_counter_ns() - t_start) * 1e9
    print(f'# Images/Second: {n_images_per_second:.0f}')

def set_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("GPU is not enabled in this notebook.")
    else:
        print("GPU is enabled in this notebook. \n")

    return device


DEVICE = set_device()
print(DEVICE)
