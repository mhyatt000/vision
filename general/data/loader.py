from general.config import cfg
from torch.utils.data import DataLoader
from .datasets import WBLOT

ds = {
    "WBLOT": WBLOT,
}


def build_loader():
    """custom dataloader"""

    return DataLoader(
        ds["WBLOT"](),
        batch_size=cfg.LOADER.BS or 64,
        shuffle=cfg.LOADER.SHUFFLE or True,
    )
