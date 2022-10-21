import datetime
import logging
import sys
import os
import math
import time

import torch
import torch.distributed as dist

""" from general.utils.comm import (
    get_world_size, all_gather, is_main_process, broadcast_data, get_rank,
) """

"what do these do??"
# from general.utils.metric_logger import MetricLogger
# from general.utils.ema import ModelEma
# from general.utils.amp import autocast, GradScaler
# from general.data.datasets.evaluation import evaluate
# from .inference import inference
# import pdb

def train():
    """...note
    why cant you just import the same cfg here?
    even if it has been modified
    """



    for iteration, (
        images,
        targets,
        idxs,
        positive_map,
        positive_map_eval,
        greenlight_map,
    ) in enumerate(data_loader, start_iter):
        nnegative = sum(len(target) < 1 for target in targets)
        nsample = len(targets)
        if nsample == nnegative or nnegative > nsample * cfg.SOLVER.MAX_NEG_PER_BATCH:
            logger.info(
                "[WARNING] Sampled {} negative in {} in a batch, greater the allowed ratio {}, skip".format(
                    nnegative, nsample, cfg.SOLVER.MAX_NEG_PER_BATCH
                )
            )
            continue

