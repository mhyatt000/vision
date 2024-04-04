import os
import os.path as osp
from abc import ABC, abstractproperty
from os.path import join

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchdata.datapipes as DP
import torchvision
import torchvision.transforms.functional as F
from PIL import Image, ImageDraw
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image
from torchvision.transforms import v2
from torchvision.utils import make_grid

""" from Mandelli et al. "Forensic Analysis of Synthetic Western Blots"
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
import cv2
from albumentations import pytorch as AP

up125 = A.Sequential(
    [
        A.Affine(scale=1.25, p=1.0),
        A.RandomCrop(256, 256, p=1.0),
    ]
)
up150 = A.Sequential(
    [
        A.Affine(scale=1.5, p=1.0),
        A.RandomCrop(256, 256, p=1.0),
    ]
)

downup50 = A.Sequential(
    [
        A.Affine(scale=0.5, p=1.0),
        A.Affine(scale=2, p=1.0),
    ]
)
downup75 = A.Sequential(
    [
        A.Affine(scale=0.75, p=1.0),
        A.Affine(scale=1.333, p=1.0),
    ]
)
downup90 = A.Sequential(
    [
        A.Affine(scale=0.9, p=1.0),
        A.Affine(scale=1.111, p=1.0),
    ]
)

jpeg70 = A.ImageCompression(70, 70, p=1.0)
jpeg80 = A.ImageCompression(80, 80, p=1.0)
jpeg90 = A.ImageCompression(90, 90, p=1.0)
jpeg100 = A.ImageCompression(100, 100, p=1.0)

jpeg = A.Compose(
    [
        A.OneOf(
            [jpeg70, jpeg80, jpeg90, jpeg100],
            p=1,
        ),
    ]
)


nothing = A.NoOp()

anyof = A.Compose(
    [
        A.OneOf(
            [
                up125,
                up150,
                downup50,
                downup75,
                downup90,
                jpeg70,
                jpeg80,
                jpeg90,
                jpeg100,
                nothing,
            ],
            p=1,
        ),
    ]
)

options = {
    "UP125": up125,
    "UP150": up150,
    "DOWNUP50": downup50,
    "DOWNUP75": downup75,
    "DOWNUP90": downup90,
    "JPEG": jpeg,
    "JPEG70": jpeg70,
    "JPEG80": jpeg80,
    "JPEG90": jpeg90,
    "JPEG100": jpeg100,
    "any": anyof,
    "nothing": nothing,
}


"""
def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

def showtens(x,a):
    grid = make_grid([x])
    show(grid)
    plt.savefig(f'blot{a}.png')
"""

flip = v2.Compose(
    [
        v2.ToTensor(),
        v2.RandomHorizontalFlip(),
        v2.RandomVerticalFlip(),
        # v2.RandomRotation(degrees=(-90, 90)),
    ]
)

normalize = v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

default = v2.Compose(
    [
        v2.Compose([v2.ToTensor(), v2.ToDtype(torch.float32, scale=True)]),
        flip,
        normalize,
    ]
)


class WBLOT(Dataset):
    """synthetic western blots dataset"""

    def __init__(
        self,
        cfg,
        root="wblot",
        transform=default,
        target_transform=None,
    ):
        super(WBLOT, self).__init__()

        self.cfg = cfg
        self.root = osp.expanduser(
            osp.join(cfg.util.machine.data, cfg.loader.data.name)
        )
        print(f"WBLOT root: {self.root}")

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

        self.classes = [x.split("/")[-1] for x in self.datafolders]

        # init labels
        self.data = []
        for i, df in enumerate(self.datafolders):
            self.data += [(img, i) for img in os.listdir(df)]

        # init transforms
        self.transform = transform
        self.target_transform = target_transform

        if cfg.loader.augment.flag:
            self.set_augment(cfg.loader.augment.train)

    def __len__(self):
        return len(self.data)

    def set_augment(self, aug):
        print(f"set WBLOT augment to: {aug}")
        self._augment = A.Compose(
            [
                options[aug],
                AP.ToTensorV2(),
            ]
        )

    def augment(self, x):
        return self._augment(image=x)["image"]

    def __getitem__(self, idx):
        rel_path, label = self.data[idx]
        img_path = join(self.datafolders[label], rel_path)

        if self.cfg.loader.augment.flag:  # albumentations needs cv2
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = read_image(img_path)

        out = {"image": torch.Tensor(image)}

        # TODO label = torch.Tensor() ... if CE then label = F.one_hot(label)
        if self.cfg.loss.body in ["ARC", "PFC"]:
            label = torch.Tensor([label])
        else:
            nclasses = len(self.datafolders)
            label = torch.Tensor([int(i == label) for i in range(nclasses)])

        if self.cfg.loader.augment.flag:
            image = self.augment(image)
        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        out.update(
            {
                "x": image,
                "label": label,
                "path": img_path,
            }
        )
        return out
        """
        if self.cfg.util.machine.device == "cuda":
            return image.pin_memory(), label.pin_memory()
        else:
            return out
        """


class SplitMixin(ABC):
    @abstractproperty
    def cfg(self):
        pass

    @abstractproperty
    def data(self):
        pass

    def split(self):
        if self.cfg.loader.split:
            idxs = list(range(len(self.data)))
            idxa, idxb = train_test_split(
                idxs, test_size=self.cfg.loader.split, random_state=self.cfg.exp.seed
            )
            return {"train": idxa, "test": idxb}


class AugmenterWrapper(DataLoader):
    def __init__(self, loader, *args, **kwargs):
        super(AugmenterWrapper, self).__init__()

        self.loader = loader
        self._augment = augment

    def set_augment(self, aug):
        print(f"set WBLOT augment to: {aug}")
        self._augment = A.Compose([options[aug], AP.ToTensorV2()])

    def augment(self, x):
        return self._augment(x)

    def __iter__(self):
        return ((x.augment(), y) for x, y in self.loader.__iter__())


class PinMemoryWrapper(DataLoader):
    def __init__(self, loader):
        super(PinMemoryWrapper, self).__init__()

        self.loader = loader
        self.pin_memory = True

    def __iter__(self):
        return ((x.pin_memory(), y.pin_memory()) for x, y in self.loader.__iter__())


@hydra.main(config_path="../../../config", config_name="main")
def main(cfg):
    # torch 1.0
    w = WBLOT(cfg)
    out = next(iter(w))

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    plt.suptitle("/".join(out["path"].split("/")[-2:]))
    for ax, im in zip(axs, ["image", "x"]):
        ax.imshow(out[im].permute(1, 2, 0).view(256, 256, 3))
        ax.set_title(im)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        ax.grid(False)

    plt.tight_layout()
    plt.show()
    quit()


if __name__ == "__main__":
    main()
