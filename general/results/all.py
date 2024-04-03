import json
import os
import warnings
from statistics import mean, variance

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.metrics import RocCurveDisplay, roc_auc_score
from sklearn.neighbors import RadiusNeighborsClassifier

from general.data.datasets import WBLOT
from general.results import out

warnings.filterwarnings("ignore")


def calc_centers(Y, Yh):
    """makes cls centers from training set"""

    norm = torch.linalg.norm
    to_rad = lambda a, b: torch.acos(torch.dot(a, b) / (norm(a) * norm(b)))
    angle = lambda a, b: (to_rad(a, b) * 180 / 3.141592)

    C = set(range(cfg.LOADER.NCLASSES))
    Y = Y.view(-1)

    centers = [torch.mean(Yh[(Y == c).view(-1)], 0) for c in C]
    return centers


def _RKNN(Y, Yh):
    """return RKNN for confusion matrix"""

    rknns = dict()
    radii = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]  # , 5e-3, 1e-2, 5e-2, 1e-1, 5e-1]
    # radii += [0.1, 0.2, 0.3, 0.4, 0.5]
    for r in radii:
        try:
            rknn = RadiusNeighborsClassifier(
                radius=r, metric="cosine", algorithm="brute", outlier_label=5
            )
            rknn.fit(Yh.cpu(), [int(x) for x in Y.cpu()])
            rknns[r] = rknn
        except:
            pass
    return rknns


def show_pca(Y, Yh, *args, centers, **kwargs):
    """docstring"""

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    ncomponents = 3  # 2
    pca = PCA(n_components=ncomponents, random_state=cfg.SOLVER.SEED)  # could do 3 dim
    Yh = pca.fit_transform(Yh.numpy(), Y.numpy())
    centers = pca.transform(centers)

    norm = lambda x: F.normalize(torch.from_numpy(x)).numpy()
    Yh, centers = norm(Yh), norm(centers)

    Y = Y.view(-1).tolist()
    labels = [CLASSES[int(y)] for y in Y]
    make_sphere(ax)
    make_centers(ax, centers)
    scatter = ax.scatter(
        Yh[:, 0], Yh[:, 1], Yh[:, 2], c=Y, label=labels, s=20
    )  # , alpha=0.3)
    # ax.view_init(0, 180)
    # plt.legend(*scatter.legend_elements())

    ticks = np.linspace(-1, 1, 9)
    labels = [""] * 9

    plt.xticks(ticks, labels)
    plt.yticks(ticks, labels)
    ax.set_zticks(ticks, labels)

    plt.tight_layout()
    mkfig("pca.png")
