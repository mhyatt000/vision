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
from general.results import  out
from general.toolbox import gpu, tqdm

from .plotter import Plotter


class CAMPlotter(Plotter):
    def __init__(self, cfg, classes=None):
        super().__init__(cfg, classes)

    def mk_layers(self, model):
        # hard coded for now

        if self.cfg.model.body == "ffc":
            main_layers = [model.layer1, model.layer2, model.layer3, model.layer4]

            real = lambda x: not isinstance(x, nn.Identity)
            global_layers = [[y.relu_g for y in x if real(y)] for x in main_layers]
            local_layers = [[y.relu_l for y in x if real(y)] for x in main_layers]

            # global_layers = sum(global_layers, [])
            local_layers = sum(local_layers, [])
            all_layers = [global_layers + local_layers]

            self.layer_groups = [global_layers, local_layers]
            self.layer_groups = global_layers

        elif self.cfg.model.body == "srnet":
            self.layer_groups = [model.block1, model.block2, model.block3, model.block4]
            self.layer_groups += [self.layer_groups]

        elif "resnet" in self.cfg.model.body :
            self.layer_groups = [model.layer1, model.layer2, model.layer3, model.layer0]
            self.layer_groups += [self.layer_groups]


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


    def calc(self, *args, model=None, testloader=None, **kwargs):
        self.mk_layers(model)

        try:
            path = osp.join(out.get_path(self.cfg),'cam')
            os.mkdir(path)
        except FileExistsError as ex:
            print(ex)
            print('cam folder already exists')

        for c in self.classes:
            try:
                os.mkdir(osp.join(path, c))
            except:
                print(f"{c} already exists")

        for layer in self.layer_groups:
            allY, allgcam, allX = [], [], []

            # Construct the CAM object once, and then re-use it on many images
            cam = GradCAM(model=model, target_layers=layer)

            @tqdm.prog(self.cfg, len(testloader), desc="CAM")
            def _cam(batch):
                todev = lambda a: a.to(self.cfg.rank, non_blocking=True)

                x = todev(batch['x'])
                label = todev(batch['label'])

                G = cam(input_tensor=X, eigen_smooth=False)

                for i in range(X.size(0)):
                    image = X[i]
                    gcam = G[i]
                    label = torch.argmax(Y[i].cpu(), dim=0)

                    image_np = np.transpose(image.cpu().numpy() , (1, 2, 0))
                    visualization = show_cam_on_image(image_np, gcam, use_rgb=True)

                    fname = out['path'][i] 

                    # Plotting
                    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
                    axs[0].imshow(image_np)
                    axs[0].set_title("Original Image")
                    axs[1].imshow(visualization)
                    axs[1].set_title("CAM Visualization")
                    axs[2].imshow(gcam, cmap="jet", alpha=0.5)
                    axs[2].set_title(f"CAM Heatmap: {fname}")

                    for ax in axs:
                        ax.axis("off")  # Turn off axis

                    fname = osp.join("cam", self.classes[label], fname)
                    self.mkfig(fname)

            for batch in testloader:
                # batch, PATH = i.items()
                _cam(batch)

    def show(self, *args, **kwargs):
        """images are already plotted in calc because it is memory intensive"""
        pass
