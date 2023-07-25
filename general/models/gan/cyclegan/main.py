from argparse import ArgumentParser as AP
import warnings

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F
import torchvision
import torchvision.utils as vutils
from torchvision import transforms as T
from tqdm import tqdm

warnings.filterwarnings("ignore")


def get_args():
    """docstring"""

    ap = AP()
    ap.add_argument("-l", "--load", type=str, help="dir load weights")
    ap.add_argument("-s", "--save", type=str, help="dir save weights")
    ap.add_argument("-t", "--train", action="store_true")
    ap.add_argument("-i", "--inference", action="store_true")

    ap.add_argument("-a", "--dir_a", type=str)
    ap.add_argument("-b", "--dir_b", type=str)

    args = ap.parse_args()
    return args


def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """Plot a list of images."""

    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        ax.set_title(titles[i]) if titles else None
    return axes


def imshow(Xh, size=(20, 10)):
    """shows fake images"""

    args = get_args()

    Xh = Xh.detach().cpu()
    fix, ax = plt.subplots(figsize=size) if size else plt.subplots()
    plt.imshow(
        vutils.make_grid(Xh, nrow=int(len(Xh) ** 0.5), padding=2, normalize=True).permute(1, 2, 0)
    )
    plt.axis("off")
    # plt.show()
    plt.pause(1)


class CycleGAN:
    """a cyclegan"""

    def __init__(self):
        pass


norm_layer = nn.InstanceNorm2d


class ResBlock(nn.Module):
    def __init__(self, f):
        super(ResBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(f, f, 3, 1, 1),
            norm_layer(f),
            nn.ReLU(),
            nn.Conv2d(f, f, 3, 1, 1),
        )
        self.norm = norm_layer(f)

    def forward(self, x):
        return F.relu(self.norm(self.conv(x) + x))


class Generator(nn.Module):
    def __init__(self, f=64, blocks=9):
        super(Generator, self).__init__()

        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, f, 7, 1, 0),
            norm_layer(f),
            nn.ReLU(True),
            nn.Conv2d(f, 2 * f, 3, 2, 1),
            norm_layer(2 * f),
            nn.ReLU(True),
            nn.Conv2d(2 * f, 4 * f, 3, 2, 1),
            norm_layer(4 * f),
            nn.ReLU(True),
        ]

        for i in range(int(blocks)):
            layers.append(ResBlock(4 * f))

        layers.extend(
            [
                nn.ConvTranspose2d(4 * f, 4 * 2 * f, 3, 1, 1),
                nn.PixelShuffle(2),
                norm_layer(2 * f),
                nn.ReLU(True),
                nn.ConvTranspose2d(2 * f, 4 * f, 3, 1, 1),
                nn.PixelShuffle(2),
                norm_layer(f),
                nn.ReLU(True),
                nn.ReflectionPad2d(3),
                nn.Conv2d(f, 3, 7, 1, 0),
                nn.Tanh(),
            ]
        )
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def block(idim, odim):
            return nn.Sequential(
                nn.Conv2d(idim, odim, 3, bias=False),
                nn.InstanceNorm2d(odim* 2),
                nn.LeakyReLU(0.2, inplace=True),
            )

        ndf = 64
        self.main = nn.Sequential(
            block(3, ndf),
            block(ndf, ndf*2),
            block(ndf * 2, ndf * 4),
            block(ndf * 4, ndf * 8),
            block(ndf * 8, ndf * 8),
            block(ndf * 8, ndf * 8),
            nn.Conv2d(ndf * 8, 1, 1)
        )

    def forward(self, input):
        return self.main(input).view(-1)


def DLoss(real, fake):
    return torch.mean((real - 1) ** 2) + torch.mean(fake ** 2)


def GLoss(fake):
    return torch.mean((fake - 1) ** 2)


def train(G1, G2, D1, D2, iterA, iterB):
    """docstring"""

    models = [G1, G2, D1, D2]
    device = "cpu"
    lr, num_epochs = 0.0002, 300 # 0.0002
    args = get_args()

    cycle_crit = torch.nn.L1Loss()

    hparam = {"lr": lr, "betas": [0.5, 0.999]}
    optimizers = [torch.optim.Adam(model.parameters(), **hparam) for model in models]

    print("starting training loop")
    for epoch in tqdm(range(num_epochs), desc="epochs"):

        for (A, _), (B, _) in tqdm(zip(iterA, iterB), leave=False, total=len(iterA)):

            # A, B = A.to(device), B.to(device)

            for model in models:
                model.zero_grad()

            Bh = G1(A)
            Ah = G2(B)

            # step D1, G1
            real = D1(B)
            fake = D1(Bh)
            loss = DLoss(real, fake)
            loss += GLoss(fake)

            # step D2, G2
            real = D2(A)
            fake = D2(Ah)
            loss += DLoss(real, fake)
            loss += GLoss(fake)

            Bhh = G1(Ah)
            Ahh = G2(Bh)

            # step D1, G1
            real = D1(B)
            fake = D1(Bhh)
            loss += DLoss(real, fake)
            loss += GLoss(fake)

            # step D2, G2
            real = D2(A)
            fake = D2(Ahh)
            loss += DLoss(real, fake)
            loss += GLoss(fake)

            loss += 10*(cycle_crit(Ahh, A) + cycle_crit(Bhh, B))
            loss.backward() 

            for optim in optimizers:
                optim.step()

            # report progress
            # imsave(Xh)

            # Show the losses
            # loss_D, loss_G = metric[0] / metric[2], metric[1] / metric[2]

        if args.save:
            torch.save(G1.state_dict(), f"{args.save}/G1.pt")
            torch.save(G2.state_dict(), f"{args.save}/G2.pt")
            torch.save(D1.state_dict(), f"{args.save}/D1.pt")
            torch.save(D2.state_dict(), f"{args.save}/D2.pt")

    # print(
    # f"loss_D {loss_D:.3f}, loss_G {loss_G:.3f}, ",
    # f"{metric[2] / timer.stop():.1f} examples/sec on {str(device)}",
    # )


def inference(G1, G2, D1, D2, iterA, iterB):
    """docstring"""

    models = [G1, G2, D1, D2]
    device = "cpu"
    args = get_args()

    for (A, _), (B, _) in tqdm(zip(iterA, iterB), leave=False, total=len(iterA)):
        with torch.no_grad():

            Bh = G1(A)
            Ah = G2(B)

            Bhh = G1(Ah)
            Ahh = G2(Bh)

            imshow(A, size=None)  # , size=(100, 50))
            imshow(Bh, size=None)  # , size=(100, 50))
            imshow(Ahh, size=None)  # , size=(100, 50))
            imshow(B, size=None)  # , size=(100, 50))
            imshow(Ah, size=None)  # , size=(100, 50))
            imshow(Bhh, size=None)  # , size=(100, 50))
            quit()


def main():

    args = get_args()

    dataA = torchvision.datasets.ImageFolder(args.dir_a)
    dataB = torchvision.datasets.ImageFolder(args.dir_b)

    tform = T.Compose(
        [
            T.CenterCrop((2048, 1024)),
            T.RandomCrop((1024, 1024)),
            T.Resize((256,256)),
            T.Resize((128,128)),
            # T.GaussianBlur(kernel_size, sigma=(0.1, 2.0))
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    dataA.transform = tform
    dataB.transform = tform

    batch_size = 1
    iterA = torch.utils.data.DataLoader(dataA, batch_size=batch_size, shuffle=True)
    iterB = torch.utils.data.DataLoader(dataB, batch_size=batch_size, shuffle=True)

    device = "cpu"
    G1 = Generator().to(device)
    G2 = Generator().to(device)
    D1 = Discriminator().to(device)
    D2 = Discriminator().to(device)

    # load weights else random init
    if args.load:
        print(f"loading weights from {args.load}")
        G1.load_state_dict(torch.load(f"{args.load}/G1.pt"))
        G2.load_state_dict(torch.load(f"{args.load}/G2.pt"))
        D1.load_state_dict(torch.load(f"{args.load}/D1.pt"))
        D2.load_state_dict(torch.load(f"{args.load}/D2.pt"))
    else:
        for model in [G1, G2, D1, D2]:
            for w in model.parameters():
                nn.init.normal_(w, 0, 0.02)

    torch.set_flush_denormal(True)
    if args.train:
        train(G1, G2, D1, D2, iterA, iterB)
    if args.inference:
        inference(G1, G2, D1, D2, iterA, iterB)


if __name__ == "__main__":
    main()
