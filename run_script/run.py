""" Essentially used as entrypoint
"""

from plant_traits.model.main import main
from plant_traits.species_model.main import species_main, test_lr

if __name__ == "__main__":

    # r2_est, val_loss, train_loss, model_cpu = main()

    val_loss, train_loss, model_cpu = species_main()
    # val_loss, train_loss = test_lr()
