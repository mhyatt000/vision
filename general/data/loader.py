from general.config import cfg
import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler

from .datasets import WBLOT
from general.results.out import get_exp_version

ds = {
    "WBLOT": WBLOT,
}

def to_swap():
    version = get_exp_version()
    return version['swap'] if version else False

def leave_out():
    version = get_exp_version()
    return version['LO'] if version else None

def leave_out_collate(data):
    X = [x for x, y in data if not y in leave_out()]
    Y = [y for x, y in data if not y in leave_out()]
    missing = cfg.LOADER.GPU_BATCH_SIZE - len(Y)
    # copy randomly to fill the gaps ... it should be random cuz random sampler
    while missing:
        X = X + X[:missing]
        Y = Y + Y[:missing]
        missing = cfg.LOADER.GPU_BATCH_SIZE - len(Y)
    assert len(X) == cfg.LOADER.GPU_BATCH_SIZE, f"samples are missing... have {len(Y)}, need {missing} for total of {cfg.LOADER.GPU_BATCH_SIZE}"
    return torch.stack(X), torch.stack(Y)


def build_loaders():
    """custom dataloader"""

    print("building loader...\n")
    print(cfg.LOADER, "\n")
    print(get_exp_version())

    dataset = ds[cfg.LOADER.DATASET]()

    if cfg.LOADER.SPLIT:
        split = [0.7, 0.3] if cfg.EXP.BODY != "5x2" else [0.5, 0.5]
        split = [int(x*len(dataset)) for x in split]
        datasets = random_split(
            dataset,
            split,
        )
        if to_swap():
            datasets = datasets[::-1]
    else:
        datasets = [dataset, ds[cfg.LOADER.DATASET]()]

    collate_fn = leave_out_collate if not leave_out() is None else None
    loaders = {}
    splits = ["train", "test"]

    for dataset, split in zip(datasets, splits):
        sampler = DistributedSampler(dataset) if cfg.distributed else None
        loader = DataLoader(
            dataset,
            batch_size=cfg.LOADER.GPU_BATCH_SIZE,
            sampler=sampler,
            shuffle=(sampler == None),
            drop_last=True if split=='train' else False,
            collate_fn=collate_fn if split == "train" else None,
            num_workers=0,
        )
        loaders[split] = loader

    return loaders
