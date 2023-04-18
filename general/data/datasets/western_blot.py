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

"""
1 an upscaling post-processing, in which the images are
enlarged by factors 1.25 and 1.5, and then randomly
cropped to fit the 256 × 256 pixel resolution;

2 a down-upscaling post-processing, in which images are
downscaled by factors 0.5, 0.75 and 0.9, and then upscaled back to 
fit their original resolution of 256 × 256 pixels;

3 a JPEG compression with different quality factors (i.e., 80, 90 and 100) 
corresponding to increasing image visual
quality.
"""

import albumentations as A

upscale = A.Compose([
    A.OneOf([
        A.Affine(scale=1.25,p=1.0),
        A.Affine(scale=1.5,p=1.0),
    ],p=1),
    A.RandomCrop(256,256,p=1.0),
])

downupscale = A.Compose([
    A.OneOf([
        A.Sequential([
            A.Affine(scale=0.5,p=1.0),
            A.Affine(scale=2,p=1.0),
        ]),
        A.Sequential([
            A.Affine(scale=0.75,p=1.0),
            A.Affine(scale=1.333,p=1.0),
        ]),
        A.Sequential([
            A.Affine(scale=0.9,p=1.0),
            A.Affine(scale=1.111,p=1.0),
        ]),
    ],p=1),
])

jpeg = A.Compose([
    A.OneOf([
        A.ImageCompression(80,80,p=1.0),
        A.ImageCompression(90,90,p=1.0),
        A.ImageCompression(100,100,p=1.0),
    ],p=1),
])

"""
preprocess = {
    'upscale':upscale,
    'downupscale': downupscale,
    'jpeg': jpeg,
}
preprocess = A.Compose([ 
    preprocess[cfg.LOADER.PREPROCESS],
    lambda x: x['image'],
])
"""

transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        # lambda x: preprocess(x) if cfg.LOADER.PREPROCESS else x,
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=(-90,90)),
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

        if cfg.LOSS.BODY in ["ARC", "ANGULAR_SM"]:  
            label = torch.Tensor([label])
        else:
            nclasses = len(self.datafolders)
            label = torch.Tensor([int(i == label) for i in range(nclasses)])

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image.pin_memory(), label.pin_memory()
