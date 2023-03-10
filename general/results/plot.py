import os
from sklearn.neighbors import RadiusNeighborsClassifier
from statistics import mean, variance
import warnings

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import torch

from general.config import cfg
from general.results import out

warnings.filterwarnings("ignore")


def mkfig(fname, legend=None):

    if legend:
        plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(out.get_path(), fname))
    print(f"saved: {fname}")
    plt.close("all")


def serialize(k, v, mode="w"):
    fname = os.path.join(out.get_path(), "results.json")
    with open(fname, "r") as file:
        data = json.load(file)

    if mode == "w":
        data[k] = v
    if mode == "a":
        try:
            data[k].append(v)
        except:
            data[k] = [v]

    with open(fname, "w") as file:
        json.dump(data, file)


def show_loss(loss, lr=None, *args, **kwargs):
    """plots loss over time"""

    #   # len(loss) // cfg.SOLVER.MAX_EPOCH
    X = [i for i, _ in enumerate(loss)]
    plt.plot(X, loss, label="loss")
    if lr:
        plt.plot(X, lr, label="learning rate")

    epochs = list(range(cfg.SOLVER.MAX_EPOCH))
    # plt.xticks([i * epoch_size for i in epochs], epochs)

    mkfig("loss.png")
    serialize("losses", loss)


def show_accuracy(acc, *args, **kwargs):
    """plots accuracy over time"""
    plt.plot([i for i, _ in enumerate(acc)], acc, color="r", label="accuracy")
    mkfig("accuracy.png")
    serialize("accuracy", acc)


def calc_confusion(Y, Yh):
    """calculate confusion matrix"""

    confusion = torch.zeros((cfg.LOADER.NCLASSES, cfg.LOADER.NCLASSES))
    Y, Yh = torch.argmax(Y, 1), torch.argmax(Yh, 1)
    for y, yh in zip(Y.view(-1), Yh.view(-1)):
        confusion[y, yh] += 1

    acc = confusion.diag().sum() / confusion.sum(1).sum()
    serialize("confusion_from_cross_entropy", confusion)
    return confusion, acc


def make_centers(Y, Yh):
    """makes cls centers from training set"""

    to_rad = lambda a, b: torch.acos(
        torch.dot(a, b) / (torch.linalg.norm(a) * torch.linalg.norm(b))
    )
    angle = lambda a, b: (to_rad(a, b) * 180 / 3.141592)

    C = set(range(cfg.LOADER.NCLASSES))
    Y = Y.view(-1)

    centers = [torch.mean(Yh[(Y == c).view(-1)], 0) for c in C]
    return centers


def arc_confusion(Y, Yh, centers):
    """confusion matrix with arc embeddings"""

    to_rad = lambda a, b: torch.acos(
        torch.dot(a, b) / (torch.linalg.norm(a) * torch.linalg.norm(b))
    )
    angle = lambda a, b: (to_rad(a, b) * 180 / 3.141592)

    Yh = torch.Tensor([[angle(yh, c) for c in centers] for yh in Yh])
    # TODO: if dist is over threshold then put in other category
    # # Y = torch.argmax(Y, 1) # not needed to argmax em
    Yh = torch.argmin(Yh, 1)

    confusion = torch.zeros((cfg.LOADER.NCLASSES, cfg.LOADER.NCLASSES))
    for y, yh in zip(Y.cpu().view(-1), Yh.cpu().view(-1)):
        confusion[int(y.item()), int(yh.item())] += 1

    acc = confusion.diag().sum() / confusion.sum(1).sum()
    serialize("confusion_from_centers", confusion)
    return confusion, acc


def _RKNN(Y, Yh):
    """return RKNN for confusion matrix"""

    rknn = RadiusNeighborsClassifier( radius=0.2,  algorithm="brute")
    rknn.fit(Yh.cpu(), [int(x) for x in Y.cpu()])
    return rknn


def show_RKNN_confusion(Y, Yh, rknn, **kwargs):
    """docstring"""

    Yh = torch.Tensor(rknn.predict(Yh))
    confusion = torch.zeros((cfg.LOADER.NCLASSES)*2)

    for y, yh in zip(Y, Yh):
        print((y), (yh))
        confusion[int(y), int(yh)] += 1

    acc = confusion.diag().sum() / confusion.sum(1).sum()

    # plt.rcParams.update({"font.size": 18}) # way too big...
    plt.matshow(confusion, cmap=plt.cm.Blues, alpha=0.3)

    for i in range(confusion.shape[0]):
        for j in range(confusion.shape[1]):
            plt.text(
                x=j,
                y=i,
                s=int(confusion[i, j]),
                va="center",
                ha="center",
                size="xx-large",
            )

    # plt.title(f"Confusion Matrix")
    plt.xlabel("Predictions")
    plt.ylabel("Ground Truth")

    mkfig("rknn.png")
    serialize("confusion_from_rknn", confusion)


def show_confusion(Y, Yh, centers=None, **kwargs):
    """builds confusion matrix"""

    confusion, acc = (
        calc_confusion(Y, Yh)
        if cfg.LOSS.BODY == "CE"
        else arc_confusion(Y, Yh, centers)
    )

    # plt.rcParams.update({"font.size": 18}) # way too big...
    plt.matshow(confusion, cmap=plt.cm.Blues, alpha=0.3)

    for i in range(confusion.shape[0]):
        for j in range(confusion.shape[1]):
            plt.text(
                x=j,
                y=i,
                s=int(confusion[i, j]),
                va="center",
                ha="center",
                size="xx-large",
            )

    # plt.title(f"Confusion Matrix")
    plt.xlabel("Predictions")
    plt.ylabel("Ground Truth")
    mkfig("confusion.png", legend=False)


def show_tsne(Y, Yh, *args, **kwargs):
    """docstring"""
    # ax = fig.add_subplot(projection="3d")

    tsne = TSNE(n_components=2, random_state=cfg.SOLVER.SEED)  # could do 3 dim
    Yh = tsne.fit_transform(Yh.cpu().numpy(), Y.cpu().numpy())

    scatter = plt.scatter(Yh[:, 0], Yh[:, 1], c=Y.view(-1).tolist(), alpha=0.3)
    # ax.scatter(Yh[:,0], Yh[:,1],Yh[:,2], c=Y.view(-1).tolist())
    # ax.view_init(0, 180)
    plt.legend(*scatter.legend_elements())
    mkfig("tsne.png")


def show_pca(Y, Yh, *args, **kwargs):
    """docstring"""

    fig, ax = plt.subplots()
    ax = fig.add_subplot(projection="3d")

    ncomponents = 3  # 2
    pca = PCA(n_components=ncomponents, random_state=cfg.SOLVER.SEED)  # could do 3 dim
    Yh = pca.fit_transform(Yh.cpu().numpy(), Y.cpu().numpy())

    scatter = plt.scatter(
        *[Yh[:, i] for i in range(ncomponents)], c=Y.view(-1).tolist(), alpha=0.3
    )
    # ax.view_init(0, 180)
    plt.legend(*scatter.legend_elements())
    mkfig("pca.png")


def calc_dprime(Y, Yh):
    """measures if positive pairs are different from negative pairs"""

    phist = {i: [] for i in range(cfg.LOADER.NCLASSES)}
    nhist = {i: [] for i in range(cfg.LOADER.NCLASSES)}
    C = set(range(cfg.LOADER.NCLASSES))

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

    dprime = (2 ** 0.5 * abs(mean(pall) - mean(nall))) / (
        (variance(pall) + variance(nall)) ** 0.2
    )
    serialize("dprime", dprime)
    return pall, nall, dprime


def show_dprime(Y, Yh, *args, **kwargs):
    """docstring"""

    pall, nall, dprime = calc_dprime(Y, Yh)

    plt.hist(pall, label="positive", bins=30, alpha=0.5)
    plt.hist(nall, label="negative", bins=30, alpha=0.5)

    plt.title(f"Population Distance (d_prime={round(dprime,4)})")
    plt.xlabel("angle")
    plt.ylabel("frequency")
    mkfig("dprime.png")


PLOTS = {
    "LOSS": show_loss,
    "CONFUSION": show_confusion,
    "RKNN": show_RKNN_confusion,
    "TSNE": show_tsne,
    "PCA": show_pca,
    "DPRIME": show_dprime,
}

if __name__ == "__main__":

    Y = torch.rand((20, 5))
    Y = torch.argmax(Y, dim=-1)

    Yh = torch.rand((20, 64))

    print(Y)
    print(Yh)

    rknn = _RKNN(Y, Yh)
    acc, conf = show_RKNN_confusion(Y, Yh, rknn)

    print(acc)
    print(conf)
