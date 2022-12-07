import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler

from general.config import cfg

from .datasets import WBLOT

ds = {
    "WBLOT": WBLOT,
}

def build_loaders():
    """custom dataloader"""

    print("building loader...\n")
    print(cfg.LOADER, "\n")

    dataset = ds[cfg.LOADER.DATASET]()

    datasets = random_split(
        dataset,
        [0.7, 0.3],
        generator=torch.Generator().manual_seed(cfg.SOLVER.SEED),
    )

    loaders = {}
    splits = ["train", "test"]
    for dataset, split in zip(datasets, splits):
        sampler = DistributedSampler(dataset) if cfg.distributed else None
        loader = DataLoader(
            dataset,
            batch_size=cfg.LOADER.BATCH_SIZE,
            sampler=sampler,
            shuffle=(sampler == None),
        )
        loaders[split] = loader

    return loaders
