""" Basic training script for PyTorch """

import time
import argparse
import os
import random

import numpy as np
import torch
from torch import nn
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from general.config import cfg
from general.engine.train import do_train
from general.helpers import Trainer
from general.models import build_model

# from general.utils.comm import get_rank, synchronize
# from general.utils.dist import set_dist_print
# from general.utils.imports import import_file
# from general.utils.logger import setup_logger
# from general.utils.metric_logger import MetricLogger, TensorboardLogger
# from general.utils.miscellaneous import mkdir, save_config

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def init_seed():
    """sets random seed for experiments"""

    print(f"set random seed to {cfg.SOLVER.SEED}")
    seed = cfg.SOLVER.SEED  # + cfg.rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def ddp_init():
    """set up for ddp"""
    torch.cuda.device_count()
    if cfg.distributed:
        init_process_group(backend="nccl")
        torch.cuda.set_device(cfg.rank)


def ddp_destroy():
    """exit ddp"""
    if cfg.distributed:
        destroy_process_group()


def grad_hook(model):
    """ adds gradient clipping """

    # if cfg.MODEL.VISION.RESET_BN:
        # for name, param in model.named_buffers():
            # if "running_mean" in name:
                # nn.init.constant_(param, 0)
            # if "running_var" in name:
                # nn.init.constant_(param, 1)

    if cfg.SOLVER.GRAD_CLIP:
        clip_value = cfg.SOLVER.GRAD_CLIP
        for p in filter(lambda p: p.grad is not None, model.parameters()):
            p.register_hook(lambda grad: torch.clamp(grad, -clip_value, clip_value))

    return model


def freeze_model(model):
    """freeze model params as defined by cfg"""

    def freeze(M):
        for p in M.parameters():
            p.requires_grad = False

    if cfg.MODEL.VISION.FREEZE:
        freeze(model.backbone.body)
    if cfg.MODEL.LANGUAGE_VISION.FREEZE:
        freeze(model.language_backbone.body)
    if cfg.MODEL.FPN.FREEZE:
        freeze(model.backbone.fpn)
    if cfg.MODEL.RPN.FREEZE:
        freeze(model.rpn)


def main():

    ddp_init()
    init_seed()
    model = build_model()
    # model = freeze_model(model)
    trainer = Trainer(model)
    trainer.run()
    ddp_destroy()


if __name__ == "__main__":
    main()
