from functools import partial
import numbers
import os
import queue as Queue
import threading
from typing import Iterable

import numpy as np

from general.config import cfg
import torch
from torch import distributed
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .datasets import WBLOT
from .samplerx import DistributedSampler, get_dist_info, worker_init_fn

DATASETS = {
    "WBLOT": WBLOT,
}


transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)


def build_loaderx(num_workers=2):

    seed = cfg.SOLVER.SEED
    train_set = DATASETS[cfg.LOADER.DATASET](transform=transform)

    rank, world_size = get_dist_info()
    sampler = DistributedSampler(
        train_set, num_replicas=world_size, rank=rank, shuffle=True, seed=seed
    )

    init_fn = partial(worker_init_fn, num_workers=num_workers, rank=rank, seed=seed)

    train_loader = DataLoaderX(
        dataset=train_set,
        batch_size=cfg.LOADER.GPU_BATCH_SIZE,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=init_fn,
    )

    return train_loader


class BackgroundGenerator(threading.Thread):
    def __init__(self, generator, max_prefetch=6):
        super(BackgroundGenerator, self).__init__()

        self.queue = Queue.Queue(max_prefetch)
        self.generator = generator
        self.rank = cfg.rank
        self.daemon = True
        self.start()

    def run(self):
        torch.cuda.set_device(self.rank)
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def next(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __next__(self):
        return self.next()

    def __iter__(self):
        return self


class DataLoaderX(DataLoader):
    def __init__(self, **kwargs):
        super(DataLoaderX, self).__init__(**kwargs)

        self.rank = cfg.rank
        self.stream = torch.cuda.Stream(self.rank)

    def __iter__(self):
        self.iter = super(DataLoaderX, self).__iter__()
        self.iter = BackgroundGenerator(self.iter, self.rank)
        self.preload()
        return self

    def preload(self):
        self.batch = next(self.iter, None)
        if self.batch is None:
            return None
        with torch.cuda.stream(self.stream):
            for k in range(len(self.batch)):
                self.batch[k] = self.batch[k].to(device=self.rank, non_blocking=True)

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        if batch is None:
            raise StopIteration
        self.preload()
        return batch
