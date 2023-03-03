import random
import os

import numpy as np
import torch
from torch import distributed
from tqdm import tqdm

from general.helpers.experiment import build_experiment
from general.config import cfg

distributed.init_process_group("nccl") # , cfg.world_rank, cfg.world_size)
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def setup_seed(seed, deterministic=True):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # deterministic is slower, more reproducible
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = deterministic


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
