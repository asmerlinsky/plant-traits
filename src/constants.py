TARGETS = ["X4_mean", "X11_mean", "X18_mean", "X26_mean", "X50_mean", "X3112_mean"]

LOG_TARGETS = ["X11_mean", "X18_mean", "X26_mean", "X50_mean", "X3112_mean"]

SD = ["X4_sd", "X11_sd", "X18_sd", "X26_sd", "X50_sd", "X3112_sd"]
ID = "id"
SPECIES = "species"

N_EPOCHS = 1000
BATCH_SIZE = 128
SAMPLE_SIZE = 1024
IMG_SIZE = 312

NUM_PASS = int(SAMPLE_SIZE / BATCH_SIZE)
