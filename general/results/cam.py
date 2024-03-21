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

        default_layers = [model.layer1, model.layer2, model.layer3, model.layer4]
        default_layers = [
            [y.relu_g for y in x if not isinstance(y, nn.Identity)] for x in default_layers
        ]
        print(default_layers)
        # default_layers = [x for y in default_layers for x in y]

        # default_layers = [x for x in default_layers if not isinstance(x, nn.Identity)]
        # default_layers = [sum(default_layers, [])]
        # default_layers = [default_layers]

        # target_layers = [model.layer4[-1].relu_l]
        # target_layers = [model.fc]

        # print(default_layers)


    def other(self):
            # Note: input_tensor can be a batch tensor with several images!

            # Construct the CAM object once, and then re-use it on many images:
            cam = GradCAM(model=model, target_layers=layer)

            # We have to specify the target we want to generate
            # the Class Activation Maps for.
            # If targets is None, the highest scoring category
            # will be used for every image in the batch.
            # Here we use ClassifierOutputTarget, but you can define your own custom targets
            # That are, for example, combinations of categories, or specific outputs in a non standard model.

            targets = [ClassifierOutputSoftmaxTarget(i) for i in range(5)]

            # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
            grayscale_cam = [
                cam(input_tensor=x, eigen_smooth=False, targets=[t])[0] for t in targets
            ]

            fig, ax = plt.subplots(1, 5)
            for ax, gcam in zip(ax, grayscale_cam):
                ax.imshow(gcam, cmap="jet")
            plt.show()
            continue

            # In this example grayscale_cam has only one image in the batch:
            # grayscale_cam = grayscale_cam[0, :]

            image = np.transpose(image.numpy() / 255, (1, 2, 0))
            visualization = show_cam_on_image(image, grayscale_cam, use_rgb=True)

            print(type(visualization))

            # plt plot the image and visualization
            fig, ax = plt.subplots(1, 3)
            ax[0].imshow(image)
            ax[1].imshow(visualization)
            ax[2].imshow(grayscale_cam, cmap="jet", alpha=0.5)
            plt.show()

            # You can also get the model outputs without having to re-inference
            model_outputs = cam.outputs

