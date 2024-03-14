"""General-purpose training script for image-to-image translation.  """

import time
from trainer import Pix2PixTrainer

# from data import create_dataset
# from models import create_model
# from options.train_options import TrainOptions

# from util.visualizer import Visualizer


def main():
    # TODO make Trainer() which is a factory to select the right trainer
    trainer = Pix2PixTrainer()
    trainer.run()


if __name__ == "__main__":
    main()
