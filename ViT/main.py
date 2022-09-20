"""main file"""

from argparse import ArgumentParser as AP
import warnings

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.utils as vutils
from tqdm import tqdm

import vit

warnings.filterwarnings("ignore")


def get_args():
    """docstring"""

    ap = AP()
    ap.add_argument("-l", "--load", type=str, help="dir load weights")
    ap.add_argument("-s", "--save", type=str, help="dir save weights")
    ap.add_argument("-t", "--train", action="store_true")
    ap.add_argument("-i", "--inference", action="store_true")

    args = ap.parse_args()
    return args


def plot_losses(losses):
    """plots loss for each epoch"""

    fig, ax = plt.subplots()
    ax.plot([i for i in range(len(losses))], losses)
    ax.set(xlabel="Epoch", ylabel="Loss")
    ax.savefig("loss.png")


def train(net, dloader):

    hpm, args = net.hpm, net.args
    for epoch in range(hpm["epochs"]):

        loss, losses = 0, []
        for X, Y in tqdm(dloader):

            Yh = net(X)
            l = hpm["criterion"](Yh, Y)

            hpm["optim"].zero_grad()
            l.backward()
            hpm["optim"].step()

            if args.save:
                torch.save(net.state_dict(), f"{args.save}/{net.__name__}.pt")

            loss += l.item() * X.size(0)
            print(l.item(), X.size(0))
        losses.append(loss / len(dloader.dataset))
        plot_loss(losses)


def main():

    args = get_args()
    data_dir = "/Users/matthewhyatt/cs/.datasets/"

    transformer = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(0.5, 0.5),
        ]
    )

    bs = 256 # 4096 in paper but ... for speed ... 159.52s/it 
    data = torchvision.datasets.CIFAR100(root=data_dir, download=True)
    data.transform = transformer
    dloader = DataLoader(data, batch_size=bs, shuffle=True)

    net = vit.ViT("B", 16, i=32, nclasses=100)

    # img = torch.randn(1, 3, 32, 32)
    # preds = net(img)  # (1, 1000)
    # print(preds.shape)

    if args.load:
        print(f"loading weights from {args.load}")
        net.load_state_dict(torch.load(f"{args.load}/{net.__name__}.pt"))
    else:
        for w in net.parameters():
            nn.init.normal_(w, 0, 0.02)

    hpm = {
        "bs": bs,
        "criterion": nn.CrossEntropyLoss(),
        "device": "cpu",
        "optim": torch.optim.Adam(net.parameters(), **{"lr": 0.005, "betas": [0.9, 0.999]}),
        "latent_dim": 100,
        "epochs": 200,
    }

    net = net.to(hpm["device"])
    net.args = args
    net.hpm = hpm

    if 1 or args.train:
        train(net, dloader)

    if args.inference:
        pass

    print("done")


if __name__ == "__main__":
    main()
