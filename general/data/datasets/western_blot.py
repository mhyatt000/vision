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
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision
from torchvision.io import read_image

from general.config import cfg


class WBLOT(Dataset):
    """synthetic western blots dataset"""

    def __init__( self, root="western_blots", transform=None, target_transform=None):
        super(WBLOT, self).__init__()

        # init folders
        self.root = join(cfg.DATASETS.LOC, root)

        self.real = join(self.root, "real")
        self.synth = join(self.root, "synth")

        self.cyclegan = join(self.synth, "cyclegan")
        self.ddpm = join(self.synth, "ddpm")
        self.pix2pix = join(self.synth, "pix2pix")
        self.stylegan2ada = join(self.synth, "stylegan2ada")

        self.datafolders = [self.real, self.cyclegan, self.ddpm, self.pix2pix, self.stylegan2ada]

        # init labels
        self.data = []
        for i,df in enumerate(self.datafolders):
            self.data += [(img,i) for img in os.listdir(df)]

        # init transforms
        self.transform = transform
        self.target_transform = target_transform


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        x, label = self.data[idx]
        img_path = join(self.datafolders[label],x)
        image = read_image(img_path)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label
