from matplotlib import pyplot as plt

from .all import *
from .plotter import Plotter


class EmbedPlotter(Plotter):
    def show(Y, Yh, *args, centers, **kwargs):
        """plots image embeddings"""

        fig, ax = plt.subplots(figsize=(10, 10))
        ax = fig.add_subplot(projection="3d")

        make_sphere(ax)
        if centers:
            make_centers(ax, centers)

        Y = Y.view(-1).tolist()
        labels = [CLASSES[int(y)] for y in Y]
        if Yh.shape[-1] > 3:
            Yh = F.normalize(Yh[:, :3])
        scatter = ax.scatter(
            Yh[:, 0], Yh[:, 1], Yh[:, 2], c=Y, label=labels, s=20
        )  # alpha=0.3
        # ax.view_init(0, 180)

        # plt.legend(*scatter.legend_elements())
        mkfig("embed.png")

    def calc(self):
        """Embeddings are already calculated by the model Tester"""
        pass
