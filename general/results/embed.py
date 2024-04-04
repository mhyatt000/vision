import matplotlib.pyplot as plt
import torch

from .all import *
from .plotter import Plotter


class EmbedPlotter(Plotter):

    def show(self, Y, embed, *args, centers=None, **kwargs):
        """plots image embeddings"""

        fig, ax = plt.subplots(figsize=(10, 10))
        ax = fig.add_subplot(projection="3d")

        self.make_sphere(ax)
        if centers:
            self.make_centers(ax, centers)

        # use argmax to turn one hot into class labels
        Y = torch.argmax(Y, dim=-1)
        Y = Y.view(-1).tolist()
        labels = [self.classes[int(y)] for y in Y]
        if embed.shape[-1] > 3:
            embed = F.normalize(embed[:, :3])
        scatter = ax.scatter(
            embed[:, 0], embed[:, 1], embed[:, 2], c=Y, label=labels, s=20
        )  # alpha=0.3
        # ax.view_init(0, 180)

        plt.legend(*scatter.legend_elements())
        self.mkfig("embed.png")

    def calc(self, *args, **kwargs):
        """Embeddings are already calculated by the model Tester"""
        pass

    def make_centers(self, ax, centers):
        """plot cls centers"""

        colors = plt.cm.viridis(np.linspace(0, 1, self.cfg.loader.n_classes))

        # can only plot 3d centers... not all dimensions
        if len(centers[0]) > 3:
            centers = F.normalize(centers[:, :3])

        for i, C in enumerate(centers.tolist()):
            C = [(0, c) for c in C]
            ax.plot(*C, c=colors[i], label=self.classes[i], lw=3)
        ax.legend()


    def make_sphere(self, ax):
        """plot a sphere"""

        r = 1
        pi, cos, sin = np.pi, np.cos, np.sin
        phi, theta = np.mgrid[0.0:pi:100j, 0.0 : 2.0 * pi : 100j]
        x = r * sin(phi) * cos(theta)
        y = r * sin(phi) * sin(theta)
        z = r * cos(phi)
        ax.plot_surface(x, y, z, rstride=1, cstride=1, color="w", alpha=0.3, linewidth=0)
