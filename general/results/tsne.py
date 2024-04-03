import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

from .plotter import Plotter


class TSNEPlotter(Plotter):
    """
    plotter for t-SNE
    - dim: 2 or 3
    """

    def __init__(self, cfg, classes=None, dim=2):
        super().__init__(cfg, classes=classes)
        self.dim = dim
        # is this init bad practice (brittle)

    def calc(self, Y, Yh, *args, **kwargs):
        self.tsne = TSNE(
            n_components=self.dim,
            random_state=self.cfg.exp.seed,
            n_iter=self.cfg.results.tsne.iter,
            perplexity=self.cfg.results.tsne.perplexity,
            n_jobs=-1,
            # metric="cosine",
        )

        self.Yh = self.tsne.fit_transform(Yh.numpy(), Y.numpy())

        # for sphere plotting
        # if self.dim == 3:
        # Yh = F.normalize(torch.Tensor(Yh)).numpy()

        # use argmax to turn one hot into class labels
        Y = torch.argmax(Y, dim=-1)
        self.Y = Y.view(-1).tolist()

        self.labels = [self.classes[int(y)] for y in Y]

    def show(self, *args, **kwargs):
        fig, ax = plt.subplots(figsize=(10, 10))

        if self.dim == 3:  # for 3D t-SNE
            ax = fig.add_subplot(projection="3d")
            make_sphere(ax)

        scatter = ax.scatter(
            *[self.Yh[:, i] for i in range(self.dim)],
            c=self.Y,
            alpha=0.3,
            s=20,
            label=self.labels
        )

        plt.legend(*scatter.legend_elements())
        mkfig("tsne.png")
