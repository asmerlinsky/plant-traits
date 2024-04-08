import torch


def get_variable_cols(variable, columns):
    return columns[columns.str.contains(prefix)]


def set_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("GPU is not enabled in this notebook.")
    else:
        print("GPU is enabled in this notebook. \n")

    return device


DEVICE = set_device()
print(DEVICE)
