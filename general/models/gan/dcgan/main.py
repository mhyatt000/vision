from argparse import ArgumentParser as AP
import warnings

import matplotlib.pyplot as plt
import torch
from torch import nn
import torchvision
from tqdm import tqdm

import torchvision.utils as vutils
import gan

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


def imsave(Xh, size=(20, 10), filename="Xh"):
    """saves fake images to current dir"""

    args = get_args()
    if args.save:
        Xh = Xh.detach().cpu()
        fix, ax = plt.subplots(figsize=size)
        plt.imshow(
            vutils.make_grid(Xh, nrow=int(len(Xh) ** 0.5), padding=2, normalize=True).permute(1, 2, 0)
        )
        plt.axis("off")
        plt.savefig(args.save + '/' + filename)
        plt.clf()


def train(netD, netG, data_iter, num_epochs, lr, latent_dim, device="cpu", *, args=get_args()):

    loss = nn.BCEWithLogitsLoss(reduction="sum")
    BCE = nn.BCELoss()

    netD, netG = netD.to(device), netG.to(device)

    hparam = {"lr": lr, "betas": [0.5, 0.999]}
    optimD = torch.optim.Adam(netD.parameters(), **hparam)
    optimG = torch.optim.Adam(netG.parameters(), **hparam)

    print("starting training loop")
    for epoch in range(num_epochs):
        print(f"{epoch} of {num_epochs}")

        for X, _ in tqdm(data_iter):

            # train D on true samples
            netD.zero_grad()
            X = X.to(device)
            batch_size = X.shape[0]
            Yh = netD(X).view(-1)
            Y = torch.full((batch_size,), 1, dtype=torch.float, device=device)
            errD = BCE(Yh, Y)
            errD.backward()

            # train D on fake samples
            # Z = torch.normal(0, 1, size=(batch_size, latent_dim, 1, 1), device=device)
            Z = torch.randn(batch_size, latent_dim, 1, 1, device=device)
            Xh = netG(Z)
            Y = torch.full((batch_size,), 0, dtype=torch.float, device=device)
            Yh = netD(Xh.detach()).view(-1)
            errD = BCE(Yh, Y)
            errD.backward()

            optimD.step()

            # train G
            netG.zero_grad()
            # the generator minimizes correct guesses so we use label 1 here
            Y = torch.full((batch_size,), 1, dtype=torch.float, device=device)
            Yh = netD(Xh).view(-1)
            errG = BCE(Yh, Y)
            errG.backward()

            optimG.step()

            # report progress
            imsave(Xh)

            # Show the losses
            # loss_D, loss_G = metric[0] / metric[2], metric[1] / metric[2]

            if args.save:
                torch.save(netG.state_dict(), f"{args.save}/G.pt")
                torch.save(netD.state_dict(), f"{args.save}/D.pt")

    print(
        f"loss_D {loss_D:.3f}, loss_G {loss_G:.3f}, ",
        f"{metric[2] / timer.stop():.1f} examples/sec on {str(device)}",
    )


def main():

    args = get_args()

    data_dir = "/Users/matthewhyatt/cs/.datasets/pokemon"
    pokemon = torchvision.datasets.ImageFolder(data_dir)

    batch_size = 256
    transformer = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((64, 64)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(0.5, 0.5),
        ]
    )

    pokemon.transform = transformer
    data_iter = torch.utils.data.DataLoader(pokemon, batch_size=batch_size, shuffle=True)

    # for x,y in data_iter:
    # imgs = x[0:20, :, :, :].permute(0, 2, 3, 1) / 2 + 0.5
    # show_images(imgs, num_rows=4, num_cols=5)
    # plt.show()
    # break

    device = "cpu"
    netG = gan.Generator().to(device)
    netD = gan.Discriminator().to(device)


    # load weights else random init
    if args.load:
        print(f"loading weights from {args.load}")
        netG.load_state_dict(torch.load(f"{args.load}/G.pt"))
        netD.load_state_dict(torch.load(f"{args.load}/D.pt"))
    else:
        for w in netD.parameters():
            nn.init.normal_(w, 0, 0.02)
        for w in netG.parameters():
            nn.init.normal_(w, 0, 0.02)

    latent_dim, lr, num_epochs = 100, 0.005, 30

    """TODO define a trainer object
    # Create the generator
    netG = Generator(ngpu).to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))

    netG.apply(weights_init)
    """

    if args.train:
        train(netD, netG, data_iter, num_epochs, lr, latent_dim)

    if args.inference:

        Z = torch.randn(64, latent_dim, 1, 1, device=device)
        # Z = torch.normal(0, 1, size=(64, latent_dim, 1, 1), device=device)
        Xh = netG(Z)

        imsave(Xh, size=(100, 50), filename="Xh0")

    print("done")


if __name__ == "__main__":
    main()
