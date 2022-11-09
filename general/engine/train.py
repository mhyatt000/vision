import datetime
import logging
import math
import os

# from general.utils.metric_logger import MetricLogger
# from general.utils.ema import ModelEma
# from general.utils.amp import autocast, GradScaler
# from general.data.datasets.evaluation import evaluate
# from .inference import inference
import pdb
import sys
import time

import torch
from torch import nn
import torch.distributed as dist

from general.config import cfg
from general.utils import comm

from tqdm import tqdm

conv = nn.Conv2d(768, 5, 8)
conv.to(cfg.DEVICE)


def train_iter(model, loader, trainer):

    t = tqdm(total=len(loader))
    for X, Y in loader:  # (t := tqdm(loader)):

        X, Y = X.to(cfg.DEVICE), Y.to(cfg.DEVICE)

        output = model(X)[-1]
        Yh = conv(output).view((-1, 5))

        # transform labels [1..5] to 0,-1,1?
        loss = trainer.loss(Yh, Y)
        loss.backward()
        trainer.optimizer.step()
        # trainer.scheduler.step()

        acc = sum([y.argmax() == yh.argmax() for y, yh in zip(Y, Yh)]) / len(Y.tolist())
        t.set_description(f'loss: {"%.4f" % loss} | acc: {acc}')
        t.update()


def do_train(model, loader, trainer):

    device = torch.device(cfg.MODEL.DEVICE)
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    global_rank = comm.get_rank()

    if cfg.SOLVER.CHECKPOINT_PER_EPOCH != -1 and cfg.SOLVER.MAX_EPOCH >= 1:
        checkpoint_period = len(loader) * cfg.SOLVER.CHECKPOINT_PER_EPOCH // cfg.SOLVER.MAX_EPOCH

    if global_rank <= 0 and cfg.SOLVER.MAX_EPOCH >= 1:
        print("Iter per epoch ", len(loader) // cfg.SOLVER.MAX_EPOCH)

    print("begin training loop")
    for epoch in range(cfg.SOLVER.MAX_EPOCH):
        print(f"epoch {epoch}")
        train_iter(model, loader, trainer)


def sandbox():
    pass


"""

    for iteration, (
        images,
        targets,
        idxs,
        positive_map,
        positive_map_eval,
        greenlight_map,
    ) in enumerate(loader, start_iter):



        nnegative = sum(len(target) < 1 for target in targets)
        nsample = len(targets)
        if nsample == nnegative or nnegative > nsample * cfg.SOLVER.MAX_NEG_PER_BATCH:
            logger.info(
                "[WARNING] Sampled {} negative in {} in a batch, greater the allowed ratio {}, skip".format(
                    nnegative, nsample, cfg.SOLVER.MAX_NEG_PER_BATCH
                )
            )
            continue

        data_time = time.time() - end
        iteration = iteration + 1
        arguments["iteration"] = iteration

        images = images.to(device)
        captions = None

        try:
            targets = [target.to(device) for target in targets]
            captions = [t.get_field("caption") for t in targets if "caption" in t.fields()]
        except:
            pass

        if cfg.SOLVER.USE_AMP:
            with autocast():
                if len(captions) > 0:
                    loss_dict = model(
                        images,
                        targets,
                        captions,
                        positive_map,
                        greenlight_map=greenlight_map,
                    )

                else:
                    loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            if torch.isnan(losses) or torch.isinf(losses):
                logging.error("NaN encountered, ignoring")
                losses[losses != losses] = 0
            optimizer.zero_grad()
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
        else:
            if len(captions) > 0:
                loss_dict = model(images, targets, captions, positive_map)
            else:
                loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            if torch.isnan(losses) or torch.isinf(losses):
                losses[losses != losses] = 0
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            scheduler.step()















        # Adapt the weight decay: only support multiStepLR
        if cfg.SOLVER.WEIGHT_DECAY_SCHEDULE and hasattr(scheduler, "milestones"):
            if milestone_target < len(scheduler.milestones):
                next_milestone = list(scheduler.milestones)[milestone_target]
            else:
                next_milestone = float("inf")
            if scheduler.last_epoch >= next_milestone * cfg.SOLVER.WEIGHT_DECAY_SCHEDULE_RATIO:
                gamma = scheduler.gamma
                logger.info("Drop the weight decay by {}!".format(gamma))
                for param in optimizer.param_groups:
                    if "weight_decay" in param:
                        param["weight_decay"] *= gamma
                # move the target forward
                milestone_target += 1

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(loss=losses_reduced, **loss_dict_reduced)
        if model_ema is not None:
            model_ema.update(model)
            arguments["model_ema"] = model_ema.state_dict()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)
        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % 20 == 0 or iteration == max_iter:
            # if iteration % 1 == 0 or iteration == max_iter:
            # logger.info(
            if global_rank <= 0:
                print(
                    meters.delimiter.join(
                        [
                            "eta: {eta}",
                            "iter: {iter}",
                            "{meters}",
                            "lr: {lr:.6f}",
                            "wd: {wd:.6f}",
                            "max mem: {memory:.0f}",
                        ]
                    ).format(
                        eta=eta_string,
                        iter=iteration,
                        meters=str(meters),
                        lr=optimizer.param_groups[0]["lr"],
                        wd=optimizer.param_groups[0]["weight_decay"],
                        memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                    )
                )
        if val_loader and (iteration % checkpoint_period == 0 or iteration == max_iter):
            if comm.is_main_process():
                print("Evaluating")
            eval_result = 0.0
            model.eval()
            if cfg.SOLVER.TEST_WITH_INFERENCE:
                with torch.no_grad():
                    try:
                        _model = model.module
                    except:
                        _model = model
                    _result = inference(
                        model=_model,
                        loader=val_loader,
                        dataset_name="val",
                        device=device,
                        expected_results=cfg.TEST.EXPECTED_RESULTS,
                        expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
                        output_folder=None,
                        cfg=cfg,
                        verbose=False,
                    )
                    if comm.is_main_process():
                        eval_result = _result[0].results["bbox"]["AP"]
            else:
                results_dict = {}
                cpu_device = torch.device("cpu")
                for i, batch in enumerate(val_loader):
                    images, targets, image_ids, positive_map, *_ = batch
                    with torch.no_grad():
                        images = images.to(device)
                        if positive_map is None:
                            output = model(images)
                        else:
                            captions = [
                                t.get_field("caption") for t in targets if "caption" in t.fields()
                            ]
                            output = model(images, captions, positive_map)
                        output = [o.to(cpu_device) for o in output]
                    results_dict.update(
                        {img_id: result for img_id, result in zip(image_ids, output)}
                    )
                all_predictions = comm.all_gather(results_dict)
                if comm.is_main_process():
                    predictions = {}
                    for p in all_predictions:
                        predictions.update(p)
                    predictions = [predictions[i] for i in list(sorted(predictions.keys()))]
                    eval_result, _ = evaluate(
                        val_loader.dataset,
                        predictions,
                        output_folder=None,
                        box_only=cfg.DATASETS.CLASS_AGNOSTIC,
                    )
                    if cfg.DATASETS.CLASS_AGNOSTIC:
                        eval_result = eval_result.results["box_proposal"]["AR@100"]
                    else:
                        eval_result = eval_result.results["bbox"]["AP"]
            model.train()

            if model_ema is not None and cfg.SOLVER.USE_EMA_FOR_MONITOR:
                model_ema.ema.eval()
                results_dict = {}
                cpu_device = torch.device("cpu")
                for i, batch in enumerate(val_loader):
                    images, targets, image_ids, positive_map, positive_map_eval = batch
                    with torch.no_grad():
                        images = images.to(device)
                        if positive_map is None:
                            output = model_ema.ema(images)
                        else:
                            captions = [
                                t.get_field("caption") for t in targets if "caption" in t.fields()
                            ]
                            output = model_ema.ema(images, captions, positive_map)
                        output = [o.to(cpu_device) for o in output]
                    results_dict.update(
                        {img_id: result for img_id, result in zip(image_ids, output)}
                    )
                all_predictions = comm.all_gather(results_dict)
                if comm.is_main_process():
                    predictions = {}
                    for p in all_predictions:
                        predictions.update(p)
                    predictions = [predictions[i] for i in list(sorted(predictions.keys()))]
                    eval_result, _ = evaluate(
                        val_loader.dataset,
                        predictions,
                        output_folder=None,
                        box_only=cfg.DATASETS.CLASS_AGNOSTIC,
                    )
                    if cfg.DATASETS.CLASS_AGNOSTIC:
                        eval_result = eval_result.results["box_proposal"]["AR@100"]
                    else:
                        eval_result = eval_result.results["bbox"]["AP"]

            arguments.update(eval_result=eval_result)

            if cfg.SOLVER.USE_AUTOSTEP:
                eval_result = comm.all_gather(eval_result)[0]  # comm.broadcast_data([eval_result])[0]
                # print("Rank {} eval result gathered".format(cfg.local_rank), eval_result)
                scheduler.step(eval_result)

            trainer.is_patient(eval_result)

        if iteration % checkpoint_period == 0:
            checkpointer.save("model_{:07d}".format(iteration), **arguments)
        if iteration == max_iter:
            checkpointer.save("model_final", **arguments)
            break

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )
"""
