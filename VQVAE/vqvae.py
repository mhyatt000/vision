import argparse
import multiprocessing as mp
import os

import numpy as np
from tensorboardX import SummaryWriter
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image

from datasets import MiniImagenet
from modules import VectorQuantizedVAE, to_scalar


def train(data_loader, model, optimizer, args, writer):
    for images, _ in data_loader:
        images = images.to(args.device)

        optimizer.zero_grad()
        x_tilde, z_e_x, z_q_x = model(images)

        # Reconstruction loss
        loss_recons = F.mse_loss(x_tilde, images)
        # Vector quantization objective
        loss_vq = F.mse_loss(z_q_x, z_e_x.detach())
        # Commitment objective
        loss_commit = F.mse_loss(z_e_x, z_q_x.detach())

        loss = loss_recons + loss_vq + args.beta * loss_commit
        loss.backward()

        # Logs
        writer.add_scalar("loss/train/reconstruction", loss_recons.item(), args.steps)
        writer.add_scalar("loss/train/quantization", loss_vq.item(), args.steps)

        optimizer.step()
        args.steps += 1


def test(data_loader, model, args, writer):
    with torch.no_grad():
        loss_recons, loss_vq = 0.0, 0.0
        for images, _ in data_loader:
            images = images.to(args.device)
            x_tilde, z_e_x, z_q_x = model(images)
            loss_recons += F.mse_loss(x_tilde, images)
            loss_vq += F.mse_loss(z_q_x, z_e_x)

        loss_recons /= len(data_loader)
        loss_vq /= len(data_loader)

    # Logs
    writer.add_scalar("loss/test/reconstruction", loss_recons.item(), args.steps)
    writer.add_scalar("loss/test/quantization", loss_vq.item(), args.steps)

    return loss_recons.item(), loss_vq.item()


def generate_samples(images, model, args):
    with torch.no_grad():
        images = images.to(args.device)
        x_tilde, _, _ = model(images)
    return x_tilde


def main(args):
    writer = SummaryWriter("./logs/{0}".format(args.output_folder))
    save_filename = "./models/{0}".format(args.output_folder)

    if args.dataset in ["mnist", "fashion-mnist", "cifar10"]:
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )
        if args.dataset == "mnist":
            # Define the train & test datasets
            train_dataset = datasets.MNIST(
                args.data_folder, train=True, download=True, transform=transform
            )
            test_dataset = datasets.MNIST(args.data_folder, train=False, transform=transform)
            num_channels = 1
        elif args.dataset == "fashion-mnist":
            # Define the train & test datasets
            train_dataset = datasets.FashionMNIST(
                args.data_folder, train=True, download=True, transform=transform
            )
            test_dataset = datasets.FashionMNIST(args.data_folder, train=False, transform=transform)
            num_channels = 1
        elif args.dataset == "cifar10":
            # Define the train & test datasets
            train_dataset = datasets.CIFAR10(
                args.data_folder, train=True, download=True, transform=transform
            )
            test_dataset = datasets.CIFAR10(args.data_folder, train=False, transform=transform)
            num_channels = 3
        valid_dataset = test_dataset
    elif args.dataset == "miniimagenet":
        transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(128),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        # Define the train, valid & test datasets
        train_dataset = MiniImagenet(
            args.data_folder, train=True, download=True, transform=transform
        )
        valid_dataset = MiniImagenet(
            args.data_folder, valid=True, download=True, transform=transform
        )
        test_dataset = MiniImagenet(args.data_folder, test=True, download=True, transform=transform)
        num_channels = 3

    # Define the data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=True)

    # Fixed images for Tensorboard
    fixed_images, _ = next(iter(test_loader))
    fixed_grid = make_grid(fixed_images, nrow=8, range=(-1, 1), normalize=True)
    writer.add_image("original", fixed_grid, 0)

    model = VectorQuantizedVAE(num_channels, args.hidden_size, args.k).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Generate the samples first once
    reconstruction = generate_samples(fixed_images, model, args)
    grid = make_grid(reconstruction.cpu(), nrow=8, range=(-1, 1), normalize=True)
    writer.add_image("reconstruction", grid, 0)

    best_loss = -1.0
    for epoch in range(args.num_epochs):
        train(train_loader, model, optimizer, args, writer)
        loss, _ = test(valid_loader, model, args, writer)

        reconstruction = generate_samples(fixed_images, model, args)
        grid = make_grid(reconstruction.cpu(), nrow=8, range=(-1, 1), normalize=True)
        writer.add_image("reconstruction", grid, epoch + 1)

        if (epoch == 0) or (loss < best_loss):
            best_loss = loss
            with open("{0}/best.pt".format(save_filename), "wb") as f:
                torch.save(model.state_dict(), f)
        with open("{0}/model_{1}.pt".format(save_filename, epoch + 1), "wb") as f:
            torch.save(model.state_dict(), f)


if __name__ == "__main__":

    from args import get_args
    args = get_args()

    # Create logs and models folder if they don't exist
    if not os.path.exists("./logs"):
        os.makedirs("./logs")
    if not os.path.exists("./models"):
        os.makedirs("./models")


    # Slurm
    if "SLURM_JOB_ID" in os.environ:
        args.output_folder += f'-{os.environ["SLURM_JOB_ID"]}'
    if not os.path.exists(f"./models/{args.output_folder}"):
        os.makedirs("./models/{args.output_folder}")

    main(args)
