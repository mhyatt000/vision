""" Basic training script for PyTorch """

import argparse
import os
import random

from general.config import cfg, try_to_find

from general.data import make_data_loader
# from general.engine.inference import inference
# from general.engine.trainer import do_train
# from general.models import build_model
# from general.solver import make_lr_scheduler, make_optimizer
# from general.utils.amp import GradScaler, autocast
# from general.utils.checkpoint import DetectronCheckpointer
# from general.utils.collect_env import collect_env_info
# from general.utils.comm import get_rank, synchronize
# from general.utils.dist import set_dist_print
# from general.utils.imports import import_file
# from general.utils.logger import setup_logger
# from general.utils.metric_logger import MetricLogger, TensorboardLogger
# from general.utils.miscellaneous import mkdir, save_config
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def init_model():
    """
    returns a model with bn mean and variance init
    adds gradient clipping
    """

    """ why return device """
    model = build_model()
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    if cfg.MODEL.BACKBONE.RESET_BN:
        for name, param in model.named_buffers():
            if "running_mean" in name:
                torch.nn.init.constant_(param, 0)
            if "running_var" in name:
                torch.nn.init.constant_(param, 1)

    if cfg.SOLVER.GRAD_CLIP > 0:
        clip_value = cfg.SOLVER.GRAD_CLIP
        for p in filter(lambda p: p.grad is not None, model.parameters()):
            p.register_hook(lambda grad: torch.clamp(grad, -clip_value, clip_value))

    return model, device


def init_dataloader():
    """docstring"""

    # <TODO> Sample data from resume is disabled, due to the conflict with max_epoch
    loader = make_data_loader(is_train=True, start_iter=0)

    if cfg.TEST.DURING_TRAINING or cfg.SOLVER.USE_AUTOSTEP:
        loader_val = make_data_loader(is_train=False)[0]
    else:
        loader_val = None

    return loader, loader_val


def freeze_model(model):
    """freeze model params as defined by cfg"""

    def freeze(M):
        for p in M.parameters():
            p.requires_grad = False

    if cfg.MODEL.BACKBONE.FREEZE:
        freeze(model.backbone.body)

    if cfg.MODEL.LANGUAGE_BACKBONE.FREEZE:
        freeze(model.language_backbone.body)

    if cfg.MODEL.FPN.FREEZE:
        freeze(model.backbone.fpn)

    if cfg.MODEL.RPN.FREEZE:
        freeze(model.rpn)

    # if cfg.SOLVER.PROMPT_PROBING_LEVEL != -1:
    #     if cfg.SOLVER.PROMPT_PROBING_LEVEL == 1:
    #         for p in model.parameters():
    #             p.requires_grad = False

    #         for p in model.language_backbone.body.parameters():
    #             p.requires_grad = True

    #         for name, p in model.named_parameters():
    #             if p.requires_grad:
    #                 print(name, " : Not Frozen")
    #             else:
    #                 print(name, " : Frozen")
    #     else:
    #         assert(0)


def train():

    use_tensorboard = cfg.use_tensorboard or False

    model, device = init_model()
    model = freeze_model(model)

    model.optimizer = make_optimizer(model)
    model.scheduler = make_lr_scheduler(optimizer)

    loader, loader_val = init_dataloader()

    if cfg.distributed:
        model = DDP(
            model,
            device_ids=[cfg.local_rank],
            output_device=cfg.local_rank,
            broadcast_buffers=cfg.MODEL.BACKBONE.USE_BN,
            find_unused_parameters=cfg.SOLVER.FIND_UNUSED_PARAMETERS,
        )

    arguments = {"iteration": 0}

    save_to_disk = get_rank() == 0

    # checkpointer = DetectronCheckpointer(cfg, model, optimizer, scheduler, cfg.OUTPUT_DIR, save_to_disk)
    # extra_checkpoint_data = checkpointer.load(try_to_find(cfg.MODEL.WEIGHT))
    # arguments.update(extra_checkpoint_data)

    # if use_tensorboard:
        # meters = TensorboardLogger(
            # log_dir=cfg.OUTPUT_DIR, start_iter=arguments["iteration"], delimiter="  "
        # )
    # else:
        # meters = MetricLogger(delimiter="  ")

    do_train(
        model, data_loader,
        optimizer, scheduler, # checkpointer,
        data_loaders_val, # meters,
    )

    return model


def init_seed():
    """sets random seed for experiments"""

    seed = cfg.SOLVER.SEED + cfg.local_rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def main():

    """args is part of cfg"""

    if cfg.distributed:
        import datetime

        torch.cuda.set_device(cfg.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://", timeout=datetime.timedelta(0, 7200)
        )

    """cfg doesnt change after this"""

    init_seed()

    if cfg.OUTPUT_DIR:
        mkdir(cfg.OUTPUT_DIR)

    logger = setup_logger("general", cfg.OUTPUT_DIR, get_rank())
    # logger.info(args) # doesnt rly work since args in cfg
    logger.info("Using {num_gpus} GPUs")

    logger.info("Loaded configuration file {cfg.config_file}")
    with open(cfg.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{cfg}")

    output_config_path = os.path.join(cfg.OUTPUT_DIR, "config.yml")
    logger.info(f"Saving config into: {output_config_path}")
    # save overloaded model config in the output directory
    if cfg.save_original_config:
        import shutil

        shutil.copy(cfg.config_file, os.path.join(cfg.OUTPUT_DIR, "config_original.yml"))

    save_config(cfg, output_config_path)

    model = train()


if __name__ == "__main__":
    main()
