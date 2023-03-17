import random
import os

import numpy as np
import torch
from torch import distributed
from tqdm import tqdm

from general.helpers.experiment import build_experiment, setup_seed
from general.config import cfg

# distributed.init_process_group() # mpi4py will handle processes
# distributed.init_process_group(backend="nccl",init_method="env://") # mpi4py will handle processes
distributed.init_process_group("nccl")  # , cfg.world_rank, cfg.world_size)
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def main():
    print("running master")

    setup_seed(seed=cfg.SEED, deterministic=False)
    # print(cfg.rank, cfg.world_size)
    torch.cuda.set_device(cfg.rank)

    E = build_experiment()
    E.run()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as ex:
        print(ex)
        quit()
