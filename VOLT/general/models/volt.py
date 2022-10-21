""" this will be where the model goes """

import torch
import torch.nn as nn
import torchvision

from .vswin import WindowAttention3D

""" from GLIP ... from maskrcnn_benchmark.utils.fuse_helper 
import ( ... BiAttentionBlock ) """

" do you need a 4D dyhead?? "

class VOLT(nn.Module):
    """transformer architecture
    - bert layers 
    - 3d w/sw mha
    - xmha
    """

    def __init__(self, ):

        # stuff goes here

    def forward(self, x):

