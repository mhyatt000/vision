from general.config import cfg
from general.toolbox import gpu
import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
import sklearn
from general.data import datasets 
from general.results.out import get_exp_version

ds = {
    "WBLOT": datasets.WBLOT,
    "K700": datasets.K700
}


def to_swap():
    version = get_exp_version()
    return (version["swap"] if 'swap' in version else False) if version else False


def leave_out():
    version = get_exp_version()
    return (version["LO"] if 'LO' in version else None) if version else None


def leave_out_collate(data):
    X = [x for x, y in data if not y in leave_out()]
    Y = [y for x, y in data if not y in leave_out()]
    missing = cfg.LOADER.GPU_BATCH_SIZE - len(Y)
    # copy randomly to fill the gaps ... it should be random cuz random sampler
    while missing:
        X = X + X[:missing]
        Y = Y + Y[:missing]
        missing = cfg.LOADER.GPU_BATCH_SIZE - len(Y)
    assert (
        len(X) == cfg.LOADER.GPU_BATCH_SIZE
    ), f"samples are missing... have {len(Y)}, need {missing} for total of {cfg.LOADER.GPU_BATCH_SIZE}"
    return torch.stack(X), torch.stack(Y)


def build_loaders():
    """custom dataloader"""

    print("building loader...\n")
    print(cfg.LOADER, "\n")
    print(get_exp_version())

    dataset = ds[cfg.LOADER.DATASET]()

    if cfg.LOADER.SPLIT:

        idxs = list(range(len(dataset)))
        test_size = 0.3 if cfg.EXP.BODY != "SERIES" else 0.5
        idxa, idxb = sklearn.model_selection.train_test_split(
            idxs, test_size=test_size, random_state=cfg.SEED)

        dataseta = torch.utils.data.Subset(dataset, idxa)
        datasetb = torch.utils.data.Subset(dataset, idxb)
        datasets = [dataseta, datasetb]

        if to_swap():
            datasets = datasets[::-1]
    else:
        raise
        # datasets = [dataset, ds[cfg.LOADER.DATASET]()]

    if cfg.LOADER.AUGMENT:
        datasets[0].dataset.set_augment(cfg.LOADER.TRAIN_AUGMENT)
        datasets[1].dataset.set_augment(cfg.LOADER.TEST_AUGMENT)

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
            drop_last=True if split == "train" else False,
            collate_fn=collate_fn if split == "train" else None,
            # for going fast...
            num_workers=2, # 4
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True,
            multiprocessing_context="spawn",
        )
        loaders[split] = loader

    return loaders
