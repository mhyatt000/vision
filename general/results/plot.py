import os
from general.data.datasets import WBLOT
import json
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

    with open(fname, "w") as file:
        json.dump(data, file)


def show_loss(loss, lr=None, *args, **kwargs):
    """plots loss over time"""

    fig, axs = plt.subplots(2,1)
    X = [i for i, _ in enumerate(loss)]
    axs[0].plot(X, loss, label="loss")
    axs[1].plot(X, loss, label="log loss")
    if lr:
        axs[1].plot(X, lr, label="learning rate")

    axs[0].legend()
    axs[1].legend()
    axs[1].set_yscale('log')

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


def make_centers(Y, Yh):
    """makes cls centers from training set"""

    norm = torch.linalg.norm
    to_rad = lambda a, b: torch.acos( torch.dot(a, b) / (norm(a) * norm(b)))
    angle = lambda a, b: (to_rad(a, b) * 180 / 3.141592)

    C = set(range(cfg.LOADER.NCLASSES))
    Y = Y.view(-1)

    centers = [torch.mean(Yh[(Y == c).view(-1)], 0) for c in C]
    return centers


def arc_confusion(Y, Yh, centers):
    """confusion matrix with arc embeddings"""

    norm = torch.linalg.norm
    to_rad = lambda a, b: torch.acos( torch.dot(a, b) / (norm(a) * norm(b)))
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
    to_rad = lambda a, b: torch.acos( torch.dot(a, b) / (norm(a) * norm(b)))
    angle = lambda a, b: (to_rad(a, b) * 180 / 3.141592)

    classes = [[] for i in range(len(centers))]
    embeds = [[] for i in range(len(centers))]
    nknown = cfg.LOADER.NCLASSES

    Y = Y.cpu().view(-1).tolist()
    Yh = Yh.cpu().tolist()

    while Y:
        y, yh, = Y.pop(), torch.Tensor(Yh.pop()), 
        a = torch.Tensor([angle(yh,c) if c.sum() else 360 for c in centers]) # dont want 0 centers with no values
        i = torch.argmin(a)

        # give priority to known classes
        if any([x<thresh for x in a[:nknown]]):
            classes[i].append(y)
            embeds[i].append(yh)

        # then to unknown classes
        elif any([x<thresh for x in a]):
            classes[i].append(y)
            embeds[i].append(yh)
            # recompute center for that class
            centers[i] = make_centers(torch.Tensor([0 for _ in embeds[i]]),torch.stack(embeds[i]))[0] # reuse old code

        # lastly to potential new classes
        else:
            classes.append([y])
            embeds.append([yh])
            centers.append(yh)

    confusion = torch.zeros((len(classes),len(classes) ))
    for yh, Y in enumerate(classes):
        for y in Y:
            confusion[int(y), int(yh)] += 1

    acc = confusion.diag().sum() / confusion.sum(1).sum()
    serialize("confusion_from_centers", confusion)
    return confusion, acc


def _RKNN(Y, Yh):
    """return RKNN for confusion matrix"""

    rknns = dict()
    for r in [1e-5,  5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1]:
        try:
            rknn = RadiusNeighborsClassifier(radius=r, metric='cosine', algorithm="brute", outlier_label=5)
            rknn.fit(Yh.cpu(), [int(x) for x in Y.cpu()])
            rknns[r] = rknn
        except:
            pass
    return rknns


def show_RKNN_confusion(Y, Yh, rknns, **kwargs):
    """docstring"""

    for r,rknn in rknns.items():

        pred = torch.Tensor(rknn.predict(Yh))
        ncol = max([int(x) for x in pred]+[int(x) for x in Y])+1
        confusion = torch.zeros((ncol,ncol))

        for y, yh in zip(Y, pred):
            confusion[int(y[0]), int(yh)] += 1

        acc = confusion.diag().sum() / confusion.sum(1).sum()

        _plot_confusion(confusion,acc,f'rknn_openset{r}.png')


def show_confusion(Y, Yh, centers=None, **kwargs):
    """builds confusion matrix"""

    if cfg.LOSS.BODY == "CE" :
        confusion, acc =  calc_confusion(Y, Yh) 
        _plot_confusion(confusion,acc,'confusion.png')
    else:
        confusion, acc =  arc_confusion(Y, Yh, centers)
        _plot_confusion(confusion,acc,'confusion.png')

        # for thresh in [55,60,65,70,75]:
            # confusion, acc =  arc_confusion_openset(Y, Yh, centers,thresh)
            # _plot_confusion(confusion,acc,f'confusion_openset{thresh}.png')


def _plot_confusion(confusion,acc, fname):

    plt.matshow(confusion, cmap=plt.cm.Blues, alpha=0.3)
    label_matx()

    getname = lambda x: CLASSES[x] if x < len(CLASSES) else f'unknown{x-len(CLASSES)}'
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
    plt.yticks([i for i in range(len(confusion))],[getname(i) for i in range(len(confusion))])
    plt.setp(plt.xticks()[1], rotation=30) 
    plt.setp(plt.yticks()[1], rotation=30) 
    plt.tight_layout()
    mkfig(fname, legend=False)


def show_tsne(Y, Yh, *args, **kwargs):
    """docstring"""
    # ax = fig.add_subplot(projection="3d")

    fig, ax = plt.subplots()
    n_components = 2
    if n_components == 3:
        ax = fig.add_subplot(projection="3d")
    tsne = TSNE(n_components=n_components, random_state=cfg.SOLVER.SEED)  
    Yh = tsne.fit_transform(Yh.cpu().numpy(), Y.cpu().numpy())

    scatter = plt.scatter(*[Yh[:, i] for i in range(n_components)], c=Y.view(-1).tolist(), alpha=0.3)
    # ax.view_init(0, 180)
    plt.legend(*scatter.legend_elements())
    mkfig("tsne.png")


def show_embed(Y, Yh, *args, **kwargs):
    """docstring"""

    fig, ax = plt.subplots()
    ax = fig.add_subplot(projection="3d")

    ncomponents = 3  # 2
    scatter = plt.scatter(*[Yh[:, i] for i in range(ncomponents)], c=Y.view(-1).tolist(), alpha=0.3)
    # ax.view_init(0, 180)
    plt.legend(*scatter.legend_elements())
    mkfig("embed.png")

def show_pca(Y, Yh, *args, **kwargs):
    """docstring"""

    fig, ax = plt.subplots()
    ax = fig.add_subplot(projection="3d")

    ncomponents = 3  # 2
    pca = PCA(n_components=ncomponents, random_state=cfg.SOLVER.SEED)  # could do 3 dim
    Yh = pca.fit_transform(Yh.cpu().numpy(), Y.cpu().numpy())

    scatter = plt.scatter(*[Yh[:, i] for i in range(ncomponents)], c=Y.view(-1).tolist(), alpha=0.3)
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

    dprime = (2 ** 0.5 * abs(mean(pall) - mean(nall))) / ((variance(pall) + variance(nall)) ** 0.2)
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
