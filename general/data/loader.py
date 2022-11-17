from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from general.config import cfg

from .datasets import WBLOT

ds = {
    "WBLOT": WBLOT,
}


def build_loader():
    """custom dataloader"""

    dataset = ds["WBLOT"]()
    kwargs = dict( batch_size=cfg.LOADER.BS or 64)

    if cfg.distributed:
        kwargs['sampler'] = DistributedSampler(dataset) if cfg.distributed else None
    else:
        kwargs['shuffle'] =(cfg.LOADER.SHUFFLE and not cfg.distributed) or True

    return DataLoader(dataset, **kwargs)
