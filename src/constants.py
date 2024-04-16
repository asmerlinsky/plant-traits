TARGETS = ["X4_mean", "X11_mean", "X18_mean", "X26_mean", "X50_mean", "X3112_mean"]

SD = ["X4_sd", "X11_sd", "X18_sd", "X26_sd", "X50_sd", "X3112_sd"]
ID = "id"


N_EPOCHS = 1000
BATCH_SIZE = 64
SAMPLE_SIZE = 1024
IMG_SIZE = 384

NUM_PASS = int(SAMPLE_SIZE / BATCH_SIZE)
