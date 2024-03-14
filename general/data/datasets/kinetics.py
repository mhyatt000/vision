import json
import os
import os.path as osp

from PIL import Image, ImageDraw
import av
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms
from torchvision.io import read_image
import torchvision.transforms.functional as TF
from torchvision.utils import make_grid
from tqdm import tqdm

from general.config import cfg


pipeline=[
    dict(type='DecordInit', io_backend='disk'),
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=3,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='PackActionInputs')
]



transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=(-90, 90)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)


class K700(Dataset):
    """kinetics700 dataset"""

    def __init__(
        self,
        root="k700",
        split="train",
        transform=transform,
        target_transform=None,
    ):
        super(K700, self).__init__()

        root = "kinetics700_2020/kinetics-dataset/k700-2020"
        self.root = osp.join(cfg.DATASETS.LOC, root)
        try:
            self.root = osp.join(cfg.DATASETS.LOC, root)
        except:
            self.root = root

        assert split in ["train", "val", "test"]
        self.split = split

        self.annfolder = osp.join(self.root, "annotations/deepmind/kinetics700_2020")
        self.annfile = osp.join(self.annfolder, f"{split}.csv")
        self.annotations = pd.read_csv(self.annfile)
        self.infofile = osp.join(self.annfolder, "info.json")

        self.imgfolder = osp.join(self.root, self.split)
        self.classes = {c: i for i, c in enumerate(os.listdir(self.imgfolder))}
        self.imgnames = {c: os.listdir(osp.join(self.imgfolder, c)) for c in self.classes.keys()}

        ###
        ### clean missing videos
        ###

        if "info.json" in os.listdir(self.annfolder):
            with open(self.infofile, "r") as file:
                info = json.load(file)
                missing = info["missing"]
                self.imgpaths = info["pathnames"]
            self.annotations = self.annotations[ ~self.annotations["youtube_id"].isin(missing) ]
            self.annotations = self.annotations.reset_index(drop=True)

        if False:
            missing, pathnames = [], {}
            for idx in tqdm(range(len(self.annotations))):
                label, ytid, time_start, time_end, split = self.annotations.loc[idx]
                full_name = [f for f in self.imgnames[label] if ytid in f]

                if len(full_name) == 0:
                    missing.append(ytid)
                else:
                    pathnames[ytid] = full_name[0]

            info = {"missing": missing, "pathnames": pathnames}

        with open(self.infofile, "w") as file:
            json.dump(info, file)

        # init transforms
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.annotations)

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

    def mp42tens(self, path):
        """given an mp4 path, returns a tensor"""

        container = av.open(path)
        totens = lambda frame: TF.to_tensor(Image.fromarray(frame.to_rgb().to_ndarray()))
        frames = [totens(frame) for frame in container.decode(video=0)]
        tens = torch.stack(frames)
        return tens

    def __getitem__(self, idx):
        label, ytid, time_start, time_end, split = self.annotations.loc[idx]

        print(ytid)

        vid_path = osp.join(self.imgfolder, label, self.imgpaths[ytid])
        vid = self.mp42tens(vid_path).float()
        label = F.one_hot(torch.Tensor([self.classes[label]]).long(), 700)

        if self.transform:
            vid = self.transform(vid)
        if self.target_transform:
            label = self.target_transform(label)

        return vid.pin_memory(), label.pin_memory()


if __name__ == "__main__":
    main()
