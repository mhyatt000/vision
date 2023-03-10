from general.config import cfg
import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler

from .datasets import WBLOT

ds = {
    "WBLOT": WBLOT,
}


def build_loaders():
    """custom dataloader"""

    print("building loader...\n")
    print(cfg.LOADER, "\n")

    dataset = ds[cfg.LOADER.DATASET]()

    if cfg.LOADER.SPLIT:
        split = [0.7, 0.3] if cfg.EXP.BODY != "5x2" else [0.5, 0.5]
        datasets = random_split(
            dataset,
            split,
        )
        if cfg.LOADER.SWAP:
            datasets = datasets[::-1]
    else:
        datasets = [dataset, ds[cfg.LOADER.DATASET]()]

    def leave_out_collate(data):
        X = [x for x, y in data if y != cfg.LOADER.LEAVE_OUT]
        Y = [y for x, y in data if y != cfg.LOADER.LEAVE_OUT]
        missing = cfg.LOADER.GPU_BATCH_SIZE - len(Y)
        # copy randomly to fill the gaps ... it should be random cuz random sampler
        if missing:
            X, Y = X + X[:missing], Y + Y[:missing]
        return torch.stack(X), torch.stack(Y)

    collate_fn = leave_out_collate if cfg.LOADER.LEAVE_OUT else None
    loaders = {}
    splits = ["train", "test"]

    for dataset, split in zip(datasets, splits):
        sampler = DistributedSampler(dataset) if cfg.distributed else None
        loader = DataLoader(
            dataset,
            batch_size=cfg.LOADER.GPU_BATCH_SIZE,
            sampler=sampler,
            shuffle=(sampler == None),
            drop_last=True,
            collate_fn=collate_fn if split == "train" else None,
            num_workers=0,
        )
        loaders[split] = loader

    return loaders
