from argparse import ArgumentParser as AP
import os
from torchvision.io import read_image
import warnings
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F
import torchvision
import torchvision.utils as vutils
from torchvision import transforms as T
from tqdm import tqdm
from config import cfg
from cyclegan import Generator, Discriminator


def DLoss(real, fake):
    return torch.mean((real - 1) ** 2) + torch.mean(fake**2)


def GLoss(fake):
    return torch.mean((fake - 1) ** 2)


def inference(G1, G2, D1, D2, iterA, iterB):
    """docstring"""

    models = [G1, G2, D1, D2]
    device = "cpu"
    cfg = get_cfg()

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


class FolderPairDataset(torch.utils.data.Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        self.sources = sorted(os.listdir(cfg.dir_a))
        self.targets = sorted(os.listdir(cfg.dir_b))

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, index):
        src = Image.open(os.path.join(cfg.dir_a, self.sources[index]))
        tgt = Image.open(os.path.join(cfg.dir_b, self.targets[index]))
        if self.transform:
            src = self.transform(src)
            tgt = self.transform(tgt)
        return src, tgt


def mk_loader():
    """makes a dataloader"""

    transform = T.Compose(
        [
            T.CenterCrop((2048, 1024)),
            T.RandomCrop((1024, 1024)),
            T.Resize((256, 256)),
            T.Resize((128, 128)),
            # T.GaussianBlur(kernel_size, sigma=(0.1, 2.0))
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    dataset = FolderPairDataset(transform=transform)

    batch_size = 1
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader


class CycleGanTrainer:
    """docstring"""

    def __init__(self):
        self.G1 = Generator()
        self.G2 = Generator()
        self.D1 = Discriminator()
        self.D2 = Discriminator()
        self.models = [self.G1, self.G2, self.D1, self.D2]

        self.loader = mk_loader()

        device = "cpu"
        for model in self.models:
            model = model.to(device)
            for w in model.parameters():
                nn.init.normal_(w, 0, 0.02)

        cycle_crit = torch.nn.L1Loss()

        hparam = {"lr": cfg.lr, "betas": cfg.betas}
        optimizers = [torch.optim.Adam(model.parameters(), **hparam) for model in self.models]

    def load():
        """docstring"""

        # load weights else random init
        if cfg.load:
            print(f"loading weights from {cfg.load}")
            G1.load_state_dict(torch.load(f"{cfg.load}/G1.pt"))
            G2.load_state_dict(torch.load(f"{cfg.load}/G2.pt"))
            D1.load_state_dict(torch.load(f"{cfg.load}/D1.pt"))
            D2.load_state_dict(torch.load(f"{cfg.load}/D2.pt"))
        else:
            for model in [G1, G2, D1, D2]:
                for w in model.parameters():
                    nn.init.normal_(w, 0, 0.02)

        torch.set_flush_denormal(True)
        # if cfg.train:
        # train(G1, G2, D1, D2, loaderA, loaderB)
        # if cfg.inference:
        # inference(G1, G2, D1, D2, loaderA, loaderB)

    def step(self, A, B):
        """docstring"""

        A,B = A.to(self.device), B.to(self.device)

        for model in self.models:
            model.zero_grad()

        Bh = self.G1(A)
        Ah = self.G2(B)

        # step D1, G1
        real = self.D1(B)
        fake = self.D1(Bh)
        loss = DLoss(real, fake)
        loss += GLoss(fake)

        # step D2, G2
        real = self.D2(A)
        fake = self.D2(Ah)
        loss += DLoss(real, fake)
        loss += GLoss(fake)

        Bhh = self.G1(Ah)
        Ahh = self.G2(Bh)

        # step D1, G1
        real = self.D1(B)
        fake = self.D1(Bhh)
        loss += DLoss(real, fake)
        loss += GLoss(fake)

        # step D2, G2
        real = self.D2(A)
        fake = self.D2(Ahh)
        loss += DLoss(real, fake)
        loss += GLoss(fake)

        loss += 10 * (cycle_crit(Ahh, A) + cycle_crit(Bhh, B))
        loss.backward()

        for optim in optimizers:
            optim.step()

        # report progress
        # imsave(Xh)

        # Show the losses
        # loss_D, loss_G = metric[0] / metric[2], metric[1] / metric[2]

    def epoch(self):
        """docstring"""

        for A, B in tqdm(self.loader, leave=False, total=len(self.loader)):
            self.step(A, B)

        if cfg.save:
            torch.save(G1.state_dict(), f"{cfg.save}/G1.pt")
            torch.save(G2.state_dict(), f"{cfg.save}/G2.pt")
            torch.save(D1.state_dict(), f"{cfg.save}/D1.pt")
            torch.save(D2.state_dict(), f"{cfg.save}/D2.pt")

    # print(
    # f"loss_D {loss_D:.3f}, loss_G {loss_G:.3f}, ",
    # f"{metric[2] / timer.stop():.1f} examples/sec on {str(device)}",
    # )

    def run(self):
        """docstring"""

        print("starting training loop")
        for epoch in tqdm(range(cfg.num_epochs), desc="epochs"):
            self.epoch()


def main():
    """docstring"""

    trainer = CycleGanTrainer()
    trainer.run()


if __name__ == "__main__":
    main()
