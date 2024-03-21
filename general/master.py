import os
import random
import time
from datetime import timedelta

import hydra
import numpy as np
import torch
from torch import distributed as dist
from tqdm import tqdm

from general import config
from general.helpers.experiment import build_experiment, setup_seed

# dist.init_process_group() # mpi4py will handle processes
# dist.init_process_group(backend="nccl",init_method="env://") # mpi4py will handle processes


@hydra.main(config_path="../config", config_name="main")
def main(cfg):
    config.process(cfg)
    print("CONFIG:", cfg.exp.name, "\n")

    if cfg.util.machine.dist:
        # print( "Distributed with", cfg.world_size, "nodes and", cfg.world_size, "gpus per node.",)

        time.sleep(cfg.world_rank / 32)
        print(
            f"Rank {cfg.world_rank:2d} of {cfg.world_size} online |"
            f"{cfg.rank} of {cfg.world_size/len(cfg.nodes)} on {cfg.nodename} {cfg.nodenumber}",
            force=True,
        )

        dist.init_process_group(
            backend="gloo",
            rank=cfg.world_rank,
            world_size=cfg.world_size,
            timeout=timedelta(seconds=30),
        )

        # dist.init_process_group(backend="mpi", rank=cfg.world_rank, world_size=cfg.world_size, timeout=timedelta(seconds=30))
        # dist.init_process_group(backend="nccl", rank=cfg.world_rank, world_size=cfg.world_size, timeout=timedelta(seconds=30))

        dist.barrier()  # wait for all processes to join
        torch.cuda.device(cfg.rank)

    setup_seed(cfg, deterministic=False)

    E = build_experiment(cfg)
    print("built")
    E.run()


if __name__ == "__main__":
    main()
