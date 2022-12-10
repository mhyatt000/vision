import logging
import math
import os
from os.path import join
import pdb
import random
import re
import sys

from PIL import Image, ImageDraw
import numpy as np
import sklearn

from general.config import cfg
import torch
from torch.utils.data import Dataset
import torchvision
from torchvision.io import read_image
import torchvision.transforms.functional as F


class WBLOT(Dataset):
    """synthetic western blots dataset"""

    def __init__(self, root="western_blots", transform=None, target_transform=None):
        super(WBLOT, self).__init__()

        try:
            self.root = join(cfg.DATASETS.LOC, root)
        except:
            self.root = root

        # TODO: add WBLOT to defaults
        self.binary = cfg.LOADER.WBLOT.BINARY if 'WBLOT' in cfg.LOADER else False

        self.real = join(self.root, "real")
        self.synth = join(self.root, "synth")

        self.cyclegan = join(self.synth, "cyclegan")
        self.ddpm = join(self.synth, "ddpm")
        self.pix2pix = join(self.synth, "pix2pix")
        self.stylegan2ada = join(self.synth, "stylegan2ada")

        self.datafolders = [self.real, self.cyclegan, self.ddpm, self.pix2pix, self.stylegan2ada]

        # init labels
        self.data = []
        for i, df in enumerate(self.datafolders):
            self.data += [(img, i) for img in os.listdir(df)]
        random.shuffle(self.data)

        # init transforms
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        x, label = self.data[idx]
        img_path = join(self.datafolders[label], x)
        image = read_image(img_path).float()

        label = bool(label) if self.binary else label
        if cfg.LOSS.BODY in ["PFC", "AAM"]:  # arcface loss
            label = torch.Tensor([label])
        else:
            nclasses = 2 if self.binary else len(self.datafolders)
            label = torch.Tensor([int(i == label) for i in range(nclasses)])

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label
