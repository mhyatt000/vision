from .plotter import Plotter
from .auc import AUCPlotter
from .all import *

class PlotManager(Plotter):.

    def __init__(self, cfg, classes=None):
        self.cfg = cfg
        self.classes = classes if classes is not None else [f"c{i}" for i in range(5)]

        self.plots = {
            "confusion": ConfusionPlotter(cfg, self.classes),
            "rknn": show_RKNN_confusion,
            "tsne": show_tsne,
            "pca": show_pca,
            "embed": show_embed,
            "dprime": show_dprime,
            "auc": AUCPlotter(cfg, self.classes),
        }

    def __call__(self):
        """Calculate and shows all plots."""

        for p in self.plots:
            p()
        # self.show_loss()
        # self.show_accuracy()

    def show_loss(self, loss, lr=None):
        """Plots loss over time."""
        nplots = 2 + (1 if lr else 0)
        fig, axs = plt.subplots(nplots, 1, figsize=(5 * nplots, 10))

        X = list(range(len(loss)))
        axs[0].plot(X, loss, label="loss")
        axs[1].plot(X, loss, label="log loss")
        axs[1].set_yscale("log")

        if lr:
            X = list(range(len(lr)))
            axs[2].plot(X, lr, color="red", lw=3, label="learning rate")

        for ax in axs:
            ax.grid()
            ax.legend()

        self.serialize("losses", loss)
        self.mkfig("loss.png", verbose=False)

    def show_accuracy(self, acc):
        """Plots accuracy over time."""
        plt.plot(list(range(len(acc))), acc, color="red", label="accuracy")
        self.serialize("accuracy", acc)
        self.mkfig("accuracy.png", verbose=False)

