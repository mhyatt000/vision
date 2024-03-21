import os
import statistics as stats
from os.path import expanduser

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from .plotter import Plotter


class DPrimePlotter(Plotter):
    def calc(self, Y, Yh):
        """measures if positive pairs are different from negative pairs"""

        phist = {i: [] for i in range(self.cfg.loader.data.n_classes)}
        nhist = {i: [] for i in range(self.cfg.loader.data.n_classes)}
        C = set(range(self.cfg.loader.data.n_classes))

        Y = torch.argmax(Y, dim=-1)
        Y = Y.view(-1)

        for c in C:
            pos = Yh[(Y == c).view(-1)]
            neg = Yh[(Y != c).view(-1)]

            dist = lambda a, b: (a - b).pow(2).sum(-1).sqrt()
            angle = (
                lambda a, b: torch.acos(
                    torch.dot(a, b) / (torch.linalg.norm(a) * torch.linalg.norm(b))
                )
                * 180
                / 3.141592
            )

            n = len(pos) // 2
            ppairs = [angle(i, j) for i, j in zip(pos[:n], pos[n : 2 * n])]
            npairs = [angle(i, j) for i, j in zip(pos[:n], neg[n : 2 * n])]

            phist[c] = ppairs
            nhist[c] = npairs

        pall, nall = [], []
        for c in C:
            pall += [float(x) if not torch.isnan(x) else -1 for x in phist[c][:1000]]
            nall += [float(x) for x in nhist[c][:1000]]

        dprime = (2**0.5 * abs(stats.mean(pall) - stats.mean(nall))) / (
            (stats.variance(pall) + stats.variance(nall)) ** 0.2
        )
        serialize("dprime", dprime)
        self.pall, self.nall, self.dprime = pall, nall, dprime

    def show(self, *args, **kwargs):
        """Shows dprime plot."""

        plt.hist(self.pall, label="positive", bins=30, alpha=0.5)
        plt.hist(self.nall, label="negative", bins=30, alpha=0.5)

        plt.title(f"Population Distance (d_prime={round(self.dprime,4)})")
        plt.xlabel("angle")
        plt.ylabel("frequency")
        plt.legend()
        mkfig("dprime.png")


def leave_out():
    """
    used to calculate mean d_prime (MDP)
    for all trials in 5x2 training split
    """

    home = expanduser("~")

    x = [28.69, 28.98, 29.28, 30.71, 28.16, 26.30, 28.53, 25.93, 25.51, 24.07]
    a = [1.05, 9.92, 1.42, 4.88, 14.36, 23.52, 18.49, 22.45, 8.83, 1.04]
    b = [22.38, 23.77, 2.39, 12.8, 20.85, 7.1, 3.79, 1.94, 18.98, 16.67]
    c = [17.72, 10.26, 10.46, 19.38, 23.58, 16.61, 4.64, 17.73, 16.46, 15.95]
    d = [7.48, 21.91, 14.22, 18.74, 3.62, 18.02, 1.56, 18.46, 8.24, 6.34]

    names = ["LO0", "LO1", "LO2", "LO3", "LO4"]
    means = [f"mean: {stats.mean(x):.2f}" for x in [x, a, b, c, d]]
    names = ["\n".join([x, y]) for x, y in zip(names, means)]

    fig, ax = plt.subplots()
    # ax.violinplot([a, b, c, d], showextrema=True, showmeans=True)
    bplot = ax.boxplot([x, a, b, c, d], patch_artist=True)

    colors = ["grey", "pink", "lightblue", "lightgreen", "mediumpurple"]
    for box, color in zip(bplot["boxes"], colors):
        box.set_facecolor(color)

    plt.xticks([1, 2, 3, 4, 5], names)
    ax.set(
        xlabel="Leave Out",
        ylabel="d_prime",
        title="Distribution of d_prime in 5x2 training split",
    )

    fig.tight_layout()
    plt.savefig(os.path.join(home, "figures", "MDP.png"))
