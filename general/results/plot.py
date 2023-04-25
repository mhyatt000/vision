import numpy as np
import os
from general.data.datasets import WBLOT
import json
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.metrics import RocCurveDisplay, roc_auc_score
from statistics import mean, variance
import warnings

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import torch
import torch.nn.functional as F

from general.config import cfg
from general.results import out

warnings.filterwarnings("ignore")

CLASSES = WBLOT().classes


def label_matx():
    """labels for plt matrix"""
    pass
    # plt.set_xticks([i for i in range(len(CLASSES))],CLASSES)
    # plt.set_yticks([i for i in range(len(CLASSES))],CLASSES)


def to_json(obj):
    """Convert obj to a version which can be serialized with JSON."""

    if is_json_serializable(obj):
        return obj

    if isinstance(obj, torch.Tensor) or isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {convert_json(k): convert_json(v) for k, v in obj.items()}
    if isinstance(obj, tuple):
        return (convert_json(x) for x in obj)
    if isinstance(obj, list):
        return [convert_json(x) for x in obj]

    if hasattr(obj, "__name__") and not ("lambda" in obj.__name__):
        return convert_json(obj.__name__)
    if hasattr(obj, "__dict__") and obj.__dict__:
        obj_dict = {convert_json(k): convert_json(v) for k, v in obj.__dict__.items()}
        return {str(obj): obj_dict}

    return str(obj)


def is_json_serializable(v):
    try:
        json.dumps(v)
        return True
    except:
        return False


def mkfig(fname, legend=None, verbose=True):
    if legend:
        plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(out.get_path(), fname))
    if verbose:
        print(f"saved: {fname}")
    plt.close("all")


def serialize(k, v):
    v = to_json(v)
    fname = os.path.join(out.get_path(), "results.json")
    try:
        with open(fname, "r") as file:
            data = json.load(file)
    except:
        data = {}

    data[k] = v
    with open(fname, "w") as file:
        json.dump(data, file)


def show_loss(loss, *args, lr=None, **kwargs):
    """plots loss over time"""

    nplots = 2 + (1 if lr else 0)
    fig, axs = plt.subplots(nplots, 1, figsize=(5 * nplots, 10))

    X = [i for i, _ in enumerate(loss)]
    axs[0].plot(X, loss, label="loss")
    axs[1].plot(X, loss, label="log loss")
    axs[1].set_yscale("log")
    if lr:
        axs[2].plot(
            X, [0 for x in X[: -len(lr)]] + lr, c="r", lw=3, label="learning rate"
        )  # mixed length :( can remove later
        # axs[2].set_yscale("log")

    for ax in axs:
        ax.grid()
        ax.legend()

    serialize("losses", loss)
    mkfig("loss.png", verbose=False)


def show_accuracy(acc, *args, **kwargs):
    """plots accuracy over time"""
    plt.plot([i for i, _ in enumerate(acc)], acc, color="r", label="accuracy")
    serialize("accuracy", acc)
    mkfig("accuracy.png", verbose=False)


def calc_confusion(Y, Yh):
    """calculate confusion matrix"""

    confusion = torch.zeros((cfg.LOADER.NCLASSES, cfg.LOADER.NCLASSES))
    Y, Yh = torch.argmax(Y, 1), torch.argmax(Yh, 1)
    for y, yh in zip(Y.view(-1), Yh.view(-1)):
        confusion[y, yh] += 1

    acc = confusion.diag().sum() / confusion.sum(1).sum()
    serialize("confusion_from_cross_entropy", confusion)
    return confusion, acc


def calc_centers(Y, Yh):
    """makes cls centers from training set"""

    norm = torch.linalg.norm
    to_rad = lambda a, b: torch.acos(torch.dot(a, b) / (norm(a) * norm(b)))
    angle = lambda a, b: (to_rad(a, b) * 180 / 3.141592)

    C = set(range(cfg.LOADER.NCLASSES))
    Y = Y.view(-1)

    centers = [torch.mean(Yh[(Y == c).view(-1)], 0) for c in C]
    return centers


def arc_confusion(Y, Yh, centers):
    """confusion matrix with arc embeddings"""

    norm = torch.linalg.norm
    to_rad = lambda a, b: torch.acos(torch.dot(a, b) / (norm(a) * norm(b)))
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


def arc_confusion_openset(Y, Yh, centers, thresh):
    """confusion matrix with arc embeddings"""

    norm = torch.linalg.norm
    to_rad = lambda a, b: torch.acos(torch.dot(a, b) / (norm(a) * norm(b)))
    angle = lambda a, b: (to_rad(a, b) * 180 / 3.141592)

    classes = [[] for i in range(len(centers))]
    embeds = [[] for i in range(len(centers))]
    nknown = cfg.LOADER.NCLASSES

    Y = Y.cpu().view(-1).tolist()
    Yh = Yh.cpu().tolist()

    while Y:
        (
            y,
            yh,
        ) = (
            Y.pop(),
            torch.Tensor(Yh.pop()),
        )
        a = torch.Tensor(
            [angle(yh, c) if c.sum() else 360 for c in centers]
        )  # dont want 0 centers with no values
        i = torch.argmin(a)

        # give priority to known classes
        if any([x < thresh for x in a[:nknown]]):
            classes[i].append(y)
            embeds[i].append(yh)

        # then to unknown classes
        elif any([x < thresh for x in a]):
            classes[i].append(y)
            embeds[i].append(yh)
            # recompute center for that class
            centers[i] = calc_centers(torch.Tensor([0 for _ in embeds[i]]), torch.stack(embeds[i]))[
                0
            ]  # reuse old code

        # lastly to potential new classes
        else:
            classes.append([y])
            embeds.append([yh])
            centers.append(yh)

    confusion = torch.zeros((len(classes), len(classes)))
    for yh, Y in enumerate(classes):
        for y in Y:
            confusion[int(y), int(yh)] += 1

    acc = confusion.diag().sum() / confusion.sum(1).sum()
    serialize("confusion_from_centers", confusion)
    return confusion, acc


def _RKNN(Y, Yh):
    """return RKNN for confusion matrix"""

    rknns = dict()
    radii = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1]
    radii += [0.1, 0.2, 0.3, 0.4, 0.5]
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


def trapezoid_score(x1, y1, x2, y2):
    width = abs(x2 - x1)
    height = (y1 + y2) / 2
    return width * height

def calculate_auc(points):
    points = sorted(points, key=lambda p: p[0]) # sort points by x value
    x1, y1 = points[0]
    auc = 0.0

    for x2, y2 in points[1:]:
        score = trapezoid_score(x1, y1, x2, y2)
        auc += score
        x1, y1 = x2, y2

    return auc


def show_auc(Y, Yh, *args, logits, **kwargs):

    fig, ax = plt.subplots(figsize=(10, 10))
    logits = F.softmax(logits, dim=-1)
    Y = F.one_hot(Y.view(-1).long())

    for i in range(1,cfg.LOADER.NCLASSES):

        # binary classification AUC
        # select only the rows where Y is 1 in column 0 or i
        rows = (Y[:, 0] == 1) | (Y[:, i] == 1)
        probs = logits[rows, 0].view(-1).numpy()
        ova = Y[rows, 0].view(-1).numpy()  # one-vs-all membership

        probs = logits[:, 0].view(-1).numpy()
        ova = Y[:,0].view(-1).numpy() # one vs all membership

        tprs, fprs = [], []
        threshs = [0.05 * x for x in list(range(20))]
        # threshs = list(range(-180,180,10))
        for thresh in threshs:
            npand , npnot = np.logical_and, np.logical_not

            tp = npand(probs >= thresh, ova).sum()
            fp = npand(probs >= thresh, npnot(ova)).sum()

            tn = npand(probs < thresh, npnot(ova)).sum()
            fn = npand(probs < thresh, ova).sum()

            tprs.append((tp/(tp+fn)))
            fprs.append((fp/(fp+tn)))

        points = [(x1,y1) for x1,y1 in zip(fprs,tprs)]
        auc = calculate_auc(points)
        ax.plot(fprs, tprs, label=f"{CLASSES[i]:15s} AUC:{auc:.4f}")

    ax.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    ax.legend()
    mkfig("auc.png")


def show_RKNN_confusion(Y, Yh, rknns, logits, **kwargs):
    """docstring"""

    accs = []
    fprs = []  # false pos rate
    for r, rknn in rknns.items():
        pred = torch.Tensor(rknn.predict(Yh))
        ncol = max([int(x) for x in pred] + [int(x) for x in Y]) + 1
        confusion = torch.zeros((ncol, ncol))

        for y, yh in zip(Y, pred):
            confusion[int(y[0]), int(yh)] += 1

        total = confusion.sum(1).sum()
        tp = confusion.diag().sum()
        acc = tp / total
        accs.append(acc)
        fpr = (total - tp) / total
        fprs.append(fpr)

        _plot_confusion(confusion, acc, f"rknn_openset{r}.png")

    # plot_auc(fprs, accs)


def show_confusion(Y, Yh, centers=None, **kwargs):
    """builds confusion matrix"""

    if cfg.LOSS.BODY == "CE":
        confusion, acc = calc_confusion(Y, Yh)
        _plot_confusion(confusion, acc, "confusion.png")
    else:
        confusion, acc = arc_confusion(Y, Yh, centers)
        _plot_confusion(confusion, acc, "confusion.png")

        # for thresh in [55,60,65,70,75]:
        # confusion, acc =  arc_confusion_openset(Y, Yh, centers,thresh)
        # _plot_confusion(confusion,acc,f'confusion_openset{thresh}.png')


def _plot_confusion(confusion, acc, fname):
    plt.matshow(confusion, cmap=plt.cm.Blues, alpha=0.3)
    label_matx()

    getname = lambda x: CLASSES[x] if x < len(CLASSES) else f"unknown{x-len(CLASSES)}"
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
    plt.xticks([i for i in range(len(confusion))], [getname(i) for i in range(len(confusion))])
    plt.yticks([i for i in range(len(confusion))], [getname(i) for i in range(len(confusion))])
    plt.setp(plt.xticks()[1], rotation=30)
    plt.setp(plt.yticks()[1], rotation=30)
    plt.tight_layout()
    mkfig(fname, legend=False)


def make_centers(ax, centers):
    """plot cls centers"""

    colors = plt.cm.viridis(np.linspace(0, 1, cfg.LOADER.NCLASSES))

    # compress
    if len(centers[0]) > 3:
        centers = F.normalize(centers[:, :3])

    for i, C in enumerate(centers.tolist()):
        C = [(0, c) for c in C]
        ax.plot(*C, c=colors[i], label=CLASSES[i], lw=3)
    ax.legend()


def make_sphere(ax):
    """plot a sphere"""

    r = 1
    pi, cos, sin = np.pi, np.cos, np.sin
    phi, theta = np.mgrid[0.0:pi:100j, 0.0 : 2.0 * pi : 100j]
    x = r * sin(phi) * cos(theta)
    y = r * sin(phi) * sin(theta)
    z = r * cos(phi)
    ax.plot_surface(x, y, z, rstride=1, cstride=1, color="w", alpha=0.3, linewidth=0)


def show_tsne(Y, Yh, *args, **kwargs):
    """docstring"""

    fig, ax = plt.subplots(figsize=(10, 10))
    n_components = 3
    if n_components == 3:
        ax = fig.add_subplot(projection="3d")
        make_sphere(ax)
    tsne = TSNE(
        n_components=n_components,
        random_state=cfg.SOLVER.SEED,
        metric="cosine",
        n_iter=5000,
        perplexity=100,
        verbose=1,
        n_jobs=16,
    )
    Yh = tsne.fit_transform(Yh.numpy(), Y.numpy())
    if n_components == 3:
        Yh = F.normalize(torch.Tensor(Yh)).numpy()

    Y = Y.view(-1).tolist()
    labels = [CLASSES[int(y)] for y in Y]

    scatter = ax.scatter(
        *[Yh[:, i] for i in range(n_components)], c=Y, alpha=0.3, s=20, label=labels
    )
    # ax.view_init(0, 180)
    plt.legend(*scatter.legend_elements())
    mkfig("tsne.png")


def show_embed(Y, Yh, *args, centers, **kwargs):
    """plots image embeddings"""

    fig, ax = plt.subplots(figsize=(10, 10))
    ax = fig.add_subplot(projection="3d")

    make_sphere(ax)
    make_centers(ax, centers)

    Y = Y.view(-1).tolist()
    labels = [CLASSES[int(y)] for y in Y]
    if Yh.shape[-1] > 3:
        Yh = F.normalize(Yh[:, :3])
    scatter = ax.scatter(Yh[:, 0], Yh[:, 1], Yh[:, 2], c=Y, label=labels, s=20)  # alpha=0.3
    # ax.view_init(0, 180)

    # plt.legend(*scatter.legend_elements())
    mkfig("embed.png")


def show_pca(Y, Yh, *args, centers, **kwargs):
    """docstring"""

    fig, ax = plt.subplots(figsize=(10, 10))
    ax = fig.add_subplot(projection="3d")

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
    scatter = ax.scatter(Yh[:, 0], Yh[:, 1], Yh[:, 2], c=Y, label=labels, s=20, alpha=0.3)
    # ax.view_init(0, 180)
    # plt.legend(*scatter.legend_elements())
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
            lambda a, b: torch.acos(torch.dot(a, b) / (torch.linalg.norm(a) * torch.linalg.norm(b)))
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

    dprime = (2**0.5 * abs(mean(pall) - mean(nall))) / ((variance(pall) + variance(nall)) ** 0.2)
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
    plt.legend()
    mkfig("dprime.png")


PLOTS = {
    "LOSS": show_loss,
    "CONFUSION": show_confusion,
    "RKNN": show_RKNN_confusion,
    "TSNE": show_tsne,
    "PCA": show_pca,
    "EMBED": show_embed,
    "DPRIME": show_dprime,
    "AUC": show_auc
}
