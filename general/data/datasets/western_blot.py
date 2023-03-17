import os
from os.path import join

from PIL import Image, ImageDraw

from general.config import cfg
import torch
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms
from torchvision.io import read_image
import torchvision.transforms.functional as F

transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)


class WBLOT(Dataset):
    """synthetic western blots dataset"""

    def __init__(
        self, root="western_blots", transform=transform, target_transform=None
    ):
        super(WBLOT, self).__init__()

        try:
            self.root = join(cfg.DATASETS.LOC, root)
        except:
            self.root = root

        self.real = join(self.root, "real")
        self.synth = join(self.root, "synth")

        self.cyclegan = join(self.synth, "cyclegan")
        self.ddpm = join(self.synth, "ddpm")
        self.pix2pix = join(self.synth, "pix2pix")
        self.stylegan2ada = join(self.synth, "stylegan2ada")

        self.datafolders = [
            self.real,
            self.cyclegan,
            self.ddpm,
            self.pix2pix,
            self.stylegan2ada,
        ]

        self.classes = [x.split('/')[-1] for x in self.datafolders]

        # init labels
        self.data = []
        for i, df in enumerate(self.datafolders):
            self.data += [(img, i) for img in os.listdir(df)]

        # init transforms
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        rel_path, label = self.data[idx]
        img_path = join(self.datafolders[label], rel_path)
        image = read_image(img_path).float()

        if cfg.LOSS.BODY in ["PFC", "AAM"]:  # arcface loss
            label = torch.Tensor([label])
        else:
            nclasses = len(self.datafolders)
            label = torch.Tensor([int(i == label) for i in range(nclasses)])

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image.to("cuda"), label.to("cuda")
