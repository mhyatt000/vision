import warnings

from d2l import torch as d2l
import matplotlib.pyplot as plt
import torch
from torch import nn
import torchvision
from tqdm import tqdm

import gan

warnings.filterwarnings("ignore")


def get_args():
    """docstring"""

    pass


class CycleGAN:
    """a cyclegan"""

    def __init__(self):
        pass


def main():
    """main"""

    G1 = gan.Generator()
    G2 = gan.Generator()
    D1 = gan.Discriminator()
    D2 = gan.Discriminator()


if __name__ == "__main__":
    main()
