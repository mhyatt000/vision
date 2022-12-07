""" Basic training script for PyTorch """

import argparse
import os
import random

import numpy as np
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group
from general.config import cfg
from general.engine.train import do_train
from general.models import build_model

from general.helpers import Trainer

# from general.data import make_data_loader
# from general.engine.inference import inference
# from general.utils.amp import GradScaler, autocast
# from general.utils.checkpoint import DetectronCheckpointer
# from general.utils.collect_env import collect_env_info
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


def init_model(model):
    """
    returns a model with bn mean and variance init
    adds gradient clipping
    """

    model.to(cfg.DEVICE)
    if cfg.distributed:
        model = DDP(
            model,
            device_ids=[cfg.rank],
            output_device=cfg.rank,
        )

    if cfg.MODEL.VISION.RESET_BN:
        for name, param in model.named_buffers():
            if "running_mean" in name:
                nn.init.constant_(param, 0)
            if "running_var" in name:
                nn.init.constant_(param, 1)

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


class TEMP(nn.Module):
    def __init__(self):
        super(TEMP, self).__init__()

    def forward(self, x):
        return x[-1]


def main():

    ddp_init()
    init_seed()
    model = build_model()

    if cfg.MODEL.BODY == "RESNET":
        if cfg.LOSS.BODY in ["PFC", "AAM"]:
            model = nn.Sequential(model, TEMP(), nn.Conv2d(1024, cfg.LOSS.PFC.EMBED_DIM, 16))
        elif cfg.LOSS.BODY == "CE":
            model = nn.Sequential( model, TEMP(), nn.Conv2d(1024, cfg.LOADER.NCLASSES, 16), nn.Softmax(dim=1))

        elif cfg.MODEL.BODY == "SWINT":
            model = nn.Sequential(model, TEMP(), nn.Conv2d(768, 5, 8))

    model = init_model(model)
    # model = freeze_model(model)
    model.train()
    trainer = Trainer(model)

    trainer.run()
    ddp_destroy()


if __name__ == "__main__":
    main()
