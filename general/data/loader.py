import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler

from general.config import cfg

from .datasets import WBLOT

ds = {
    "WBLOT": WBLOT,
}


"""
SPLIT could be 0,1,2 for train,val,test
data = split(cfg.LOADER.SPLIT) if cfg.LOADER.USE_SPLIT else dataset
"""

def build_loader():
    """custom dataloader"""

    dataset = ds["WBLOT"]()

    split = random_split(
        dataset,
        [len(dataset) * x for x in [0.7, 0.15, 0.15]], # update torch?
        generator=torch.Generator().manual_seed(cfg.SOLVER.SEED),
    ) [0 if cfg.RECIPE.TRAIN else 1 if cfg.RECIPE.VAL else 2]

    kwargs = dict(batch_size=cfg.LOADER.BATCH_SIZE or 64)

    if cfg.distributed:
        kwargs["sampler"] = DistributedSampler(dataset) if cfg.distributed else None
    else:
        kwargs["shuffle"] = (cfg.LOADER.SHUFFLE and not cfg.distributed) or True

    return DataLoader(split, **kwargs)
