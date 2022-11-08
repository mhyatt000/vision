# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.


import os
import pandas as pd
from torchvision.io import read_image

import logging
import math
import os
import os.path
import pdb
import random
import re
import sys

from PIL import Image, ImageDraw
import numpy as np
import torch
import torch.utils.data
import torchvision

from general.config import cfg
from general.structures.bounding_box import BoxList
from general.structures.keypoint import PersonKeypoints
from general.structures.segmentation_mask import SegmentationMask


class SMIT(Dataset):
    """ spoken moments in time dataset"""

    def __init__(self, root, ann, transform=None, target_transform=None):
        super(SMIT, self).__init__()

        self.root = root
        self.ann = ann

        self.transform = transform
        self.target_transform = target_transform
        

    def __len__(self):
        return len
 
    def __getitem__():
        
