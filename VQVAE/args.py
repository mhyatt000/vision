from argparse import ArgumentParser as AP
import torch

def get_args():
    """docstring"""

    ap = argparse.ArgumentParser(description="VQ-VAE")

    # General
    ap.add_argument("--data-folder", type=str, help="name of the data folder")
    ap.add_argument( "--dataset", type=str, help="name of the dataset (mnist, fashion-mnist, cifar10, miniimagenet)")

    # Latent space
    ap.add_argument( "--hidden-size", type=int, default=256, help="size of the latent vectors (default: 256)")
    ap.add_argument( "--k", type=int, default=512, help="number of latent vectors (default: 512)")

    # Optimization
    ap.add_argument("--batch-size", type=int, default=128, help="batch size (default: 128)")
    ap.add_argument( "--num-epochs", type=int, default=100, help="number of epochs (default: 100)")
    ap.add_argument( "--lr", type=float, default=2e-4, help="learning rate for Adam optimizer (default: 2e-4)")
    ap.add_argument( "--beta", type=float, default=1.0, help="contribution of commitment loss, between 0.1 and 2.0 (default: 1.0)")

    # Miscellaneous
    ap.add_argument( "--output-folder", type=str, default="vqvae", help="name of the output folder (default: vqvae)")
    ap.add_argument( "--num-workers", type=int, default=mp.cpu_count() - 1, help="number of workers for trajectories sampling (default: { mp.cpu_count() - 1})")
    ap.add_argument( "--device", type=str, default="cpu", help="set the device (cpu or cuda, default: cpu)")

    args = ap.parse_args()

    # Device
    args.device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    args.steps = 0

    return args






