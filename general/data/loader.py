import sklearn
import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler

from general.data import datasets
from general.results.out import get_exp_version

ds = {
    "wblot": datasets.WBLOT,
    # "K700": datasets.K700
}


def to_swap(cfg):
    version = get_exp_version(cfg)
    return (version["swap"] if "swap" in version else False) if version else False


def leave_out():
    print("no leave out for now...")
    return
    version = get_exp_version(cfg)
    return (version["LO"] if "LO" in version else None) if version else None


"""
def leave_out_collate(data):
    X = [x for x, y in data if not y in leave_out()]
    Y = [y for x, y in data if not y in leave_out()]
    missing = cfg.loader.gpu_batch_size - len(Y)
    # copy randomly to fill the gaps ... it should be random cuz random sampler
    while missing:
        X = X + X[:missing]
        Y = Y + Y[:missing]
        missing = cfg.loader.gpu_batch_size - len(Y)
    assert (
        len(X) == cfg.loader.gpu_batch_size
    ), f"samples are missing... have {len(Y)}, need {missing} for total of {cfg.loader.gpu_batch_size}"
    return torch.stack(X), torch.stack(Y)
"""


def build_loaders(cfg):
    """custom dataloader"""

    print("building loader...\n")
    print(cfg.loader, "\n")
    print(get_exp_version(cfg))

    dataset = ds[cfg.loader.data.name](cfg)

    if cfg.loader.split:
        idxs = list(range(len(dataset)))
        test_size = 0.3 if cfg.exp.body != "SERIES" else 0.5
        idxa, idxb = sklearn.model_selection.train_test_split(
            idxs, test_size=test_size, random_state=cfg.exp.seed
        )

        dataseta = torch.utils.data.Subset(dataset, idxa)
        datasetb = torch.utils.data.Subset(dataset, idxb)
        datasets = [dataseta, datasetb]

        if to_swap(cfg):
            datasets = datasets[::-1]
    else:
        raise
        # datasets = [dataset, ds[cfg.loader.dataset]()]

    if cfg.loader.augment.flag:  # TODO double check this
        datasets[0].dataset.set_augment(cfg.loader.augment.train)
        datasets[1].dataset.set_augment(cfg.loader.augment.test)

    # collate_fn = leave_out_collate if not leave_out() is None else None
    loaders = {}
    splits = ["train", "test"]

    for dataset, split in zip(datasets, splits):
        sampler = DistributedSampler(dataset) if cfg.util.machine.dist else None
        loader = DataLoader(
            dataset,
            batch_size=cfg.loader.gpu_batch_size,
            sampler=sampler,
            shuffle=(sampler == None),
            drop_last=True if split == "train" else False,
            collate_fn=None,  # collate_fn if split == "train" else None,
            # for going fast...
            num_workers=2,  # 4
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True,
            multiprocessing_context="spawn",
        )
        loaders[split] = loader

    return loaders
