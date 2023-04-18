import random
from datetime import timedelta
# import deepspeed
import os
import time

# import horovod.torch as hvd

import numpy as np
import torch
from torch import distributed as dist
from tqdm import tqdm

from general.helpers.experiment import build_experiment, setup_seed
from general.config import cfg

# dist.init_process_group() # mpi4py will handle processes
# dist.init_process_group(backend="nccl",init_method="env://") # mpi4py will handle processes


def main():

    print(f"OMP_NUM_THREADS: {os.environ['OMP_NUM_THREADS']}")
    print("CONFIG:", cfg.config_file, "\n")

    time.sleep(cfg.world_rank / 32)
    print( f"Rank {cfg.world_rank:2d} of {cfg.world_size} online | {cfg.rank} of 4 on {cfg.nodename}",
        force=True,)

    # hvd.init()
    # torch.cuda.set_device(hvd.local__rank())

    # os.environ['MASTER_PORT'] = str(12355)
    # print(os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])

    # dist.init_process_group(backend="nccl", rank=cfg.world_rank, world_size=cfg.world_size, timeout=timedelta(seconds=30))

    dist.init_process_group(
        backend="gloo",
        rank=cfg.world_rank,
        world_size=cfg.world_size,
        timeout=timedelta(seconds=30),
    )
    # dist.init_process_group(backend="mpi", rank=cfg.world_rank, world_size=cfg.world_size, timeout=timedelta(seconds=30))
    # dist.init_process_group(backend="nccl", rank=cfg.world_rank, world_size=cfg.world_size, timeout=timedelta(seconds=30))

    dist.barrier()

    setup_seed(seed=cfg.SEED, deterministic=False)
    torch.cuda.device(cfg.rank)

    E = build_experiment()
    E.run()


if __name__ == "__main__":
    main()
