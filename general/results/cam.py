"""pip install grad-cam"""

import os
import os.path as osp

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as F
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, LayerCAM, ScoreCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import (
    ClassifierOutputSoftmaxTarget, ClassifierOutputTarget)
from torch import nn
from torchvision import transforms
from torchvision.io import read_image
from torchvision.models import resnet50

from general import config
from general.models import build_model

from .plotter import Plotter


class CAMPlotter(Plotter):
    def __init__(self, cfg, classes=None):
        super().__init__(cfg, classes)

        # hard coded for now
        main_layers = [model.layer1, model.layer2, model.layer3, model.layer4]

        real = lambda x: not isinstance(x, nn.Identity)
        global_layers = [[y.relu_g for y in x if not real(y)] for x in main_layers]
        local_layers = [[y.relu_l for y in x if not real(y)] for x in main_layers]

        global_layers = sum(global_layers, [])
        local_layers = sum(local_layers, [])
        all_layers = [global_layers + local_layers]

        # selects highest value when None
        targets = None  # [ClassifierOutputSoftmaxTarget(i) for i in range(5)]

    @staticmethod
    def gather(x, device):
        """simple all gather manuver"""

        _gather = [
            torch.zeros(x.shape, device=device) for _ in range(dist.get_world_size())
        ]
        dist.all_gather(_gather, x)
        return torch.cat(_gather)

    def calc(self, *args, **kwargs):
        allY, allYh = [], []

        # Construct the CAM object once, and then re-use it on many images:
        cam = GradCAM(model=model, target_layers=layer)

        # TODO: fix this so its clean in toolbox.tqdm.py
        # decorator was to fix printnode problem but its clunky
        @tqdm.prog(self.cfg, len(testloader), desc="CAM")
        def _cam(X, Y):
            X = X.to(self.cfg.rank, non_blocking=True)
            Y = Y.to(self.cfg.rank, non_blocking=True)

            Yh = self.model(X).view((Y.shape[0], -1))
            Yh = F.normalize(Yh)
            if self.cfg.util.machine.dist:
                Y, Yh = self.gather(Y, cfg.rank), self.gather(Yh, cfg.rank)
            allY.append(Y.cpu())
            allYh.append(Yh.cpu())

        for X, Y in testloader:
            _cam(X, Y)

        gcam = self.cam(input_tensor=x, eigen_smooth=False)

        image = np.transpose(image.numpy() / 255, (1, 2, 0))
        visualization = show_cam_on_image(image, gcam, use_rgb=True)

        # plt plot the image and visualization
        fig, ax = plt.subplots(1, 3)
        ax[0].imshow(image)
        ax[1].imshow(visualization)
        ax[2].imshow(gcam, cmap="jet", alpha=0.5)
        plt.show()

        # You can also get the model outputs without having to re-inference
        model_outputs = cam.outputs
