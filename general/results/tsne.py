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

    def calc(self, Y, embed, *args, **kwargs):
        self.tsne = TSNE(
            n_components=self.dim,
            random_state=self.cfg.exp.seed,
            n_iter=self.cfg.results.tsne.iter,
            perplexity=self.cfg.results.tsne.perplexity,
            n_jobs=-1,
            # metric="cosine",
        )

        self.embed = self.tsne.fit_transform(embed.numpy(), Y.numpy())

        # for sphere plotting
        # if self.dim == 3:
        # embed = F.normalize(torch.Tensor(embed)).numpy()

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
            *[self.embed[:, i] for i in range(self.dim)],
            c=self.Y,
            alpha=0.3,
            s=20,
            label=self.labels
        )
        self.nogrid(ax)

        plt.legend(*scatter.legend_elements())

        name = "_".join([f'{k}:{v}' for k,v in self.cfg.results.tsne.items()])
        self.mkfig(f"tsne_{name}.png")

    def nogrid(self, ax):

        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.grid(False)

