import os
import socket
import sys
import time

import hydra
import torch
from omegaconf import DictConfig, OmegaConf


def set_dist_print(is_master):
    setup_for_distributed(is_master)


def setup_for_distributed(is_master):
    """This function disables printing when not in master process"""
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def use_mpi():
    """
    if False:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD

        cfg.world_size = int(comm.Get_size())
        cfg.world_rank = int(comm.Get_rank())

    ppn = cfg.world_size // len(nodes) # process per node
    cfg.nodename = nodenames[MPI.Get_processor_name()]
    cfg.local_rank = cfg.world_rank % ppn
    os.environ['MASTER_ADDR'] = nodes[0]


    if True and not cfg.rank:
        print('no OMPI')
    """
    pass


def try_pbs(cfg):
    try:
        with open(os.environ["PBS_NODEFILE"], "r") as file:
            cfg.nodes = [n.strip("\n").split(".")[0] for n in file.readlines()]
        cfg.nodename = socket.gethostname()
        cfg.nodenumber = "NODE_" + str(cfg.nodes.index(cfg.nodename))
    except:  # probably using the login node :(
        pass
        cfg.nodes = ["LOGIN"]
        cfg.nodename = cfg.nodes[0]
        cfg.nodenumber = 0


def find_world(cfg):
    cfg.world_size = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    cfg.rank = int(os.environ["LOCAL_RANK"]) if "LOCAL_RANK" in os.environ else 0
    cfg.world_rank = int(os.environ["RANK"]) if "RANK" in os.environ else 0
    cfg.master = cfg.world_rank == 0

    cfg.util.machine.dist = cfg.world_size > 1 and cfg.util.machine.device != "cpu"
    if cfg.util.machine.device == "cpu":
        cfg.world_size = 1
        cfg.rank = "cpu"
        cfg.world_rank = 0
        cfg.master = True


def process(cfg):
    OmegaConf.set_struct(cfg, False)  # unfreeze the config

    # names
    cfg.exp.root = "/".join(__file__.split("/")[:-2])
    name = "_".join([cfg.model.body, cfg.loss.body, cfg.loader.data.name])
    cfg.exp.name  = f"{cfg.exp.name}_{name}" if cfg.exp.name else name 
    cfg.exp.out = os.path.join(cfg.exp.root, "experiments", cfg.exp.name)

    # set up distributed nodes
    find_world(cfg)
    try_pbs(cfg)

    if cfg.loader.gpu_batch_size is None:
        cfg.loader.gpu_batch_size = cfg.loader.batch_size // cfg.world_size

    if not cfg.util.all_print:  # all nodes print?
        set_dist_print(cfg.world_rank <= 0)

    # TODO: fix cast 1e-3 str to float
    # TODO: what was the reason for this?
    cfg.solver.optim.base_lr = float(cfg.solver.optim.base_lr)
    cfg.solver.optim.lr = cfg.solver.optim.base_lr

    print(OmegaConf.to_yaml(cfg))
    OmegaConf.set_struct(cfg, True)  # freeze the config
