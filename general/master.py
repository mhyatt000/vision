import random
import os

import numpy as np
import torch
from torch import distributed
from tqdm import tqdm

from general.helpers import Experiment
from general.models import build_model
from general.config import cfg
from general.models.backbone import ffcresnet
from general.data.loader import build_loaders

distributed.init_process_group("nccl")

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
    torch.cuda.set_device(cfg.rank)

    E = Experiment()
    E.run()

    # model._set_static_graph()

if __name__ == "__main__":
    main()
