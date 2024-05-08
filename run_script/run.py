from src.model.main import main
from src.species_model.main import species_main

if __name__ == "__main__":

    # r2_est, val_loss, train_loss, model_cpu = main()

    val_loss, train_loss, model_cpu = species_main()
