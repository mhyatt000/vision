from argparse import ArgumentParser as AP
import socket
import torch

import time
import os
import sys

from general.utils.dist import set_dist_print
from .defaults import _C as cfg

ap = AP(description="PyTorch Object Detection Training")

ap.add_argument( "--config-name", default="", metavar="FILE", help="path to config file", type=str)
ap.add_argument( "--deepspeed", action='store_true')

# ap.add_argument( "--config-file", default="", metavar="FILE", help="path to config file", type=str)
# ap.add_argument("--local_rank", type=int, default=0)
# ap.add_argument( "--skip-test", dest="skip_test", help="Do not test the final model", action="store_true",)
# ap.add_argument( "--use-tensorboard", dest="use_tensorboard", help="Use tensorboardX logger (Requires tensorboardX installed)", action="store_true", default=False,)
# ap.add_argument( "opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER,)
# ap.add_argument("--save_original_config", action="store_true")
# ap.add_argument("--disable_output_distributed", action="store_true")
# ap.add_argument("--override_output_dir", default=None)

args = ap.parse_args()
cfg.config_name = args.config_name.split('/')[-1].replace('.yaml','')
cfg.deepspeed = args.deepspeed
# for k,v in args.__dict__.items():
# setattr(cfg,k,v)

cfg.config_name = cfg.config_name or input("config file: ")
cfg.ROOT = "/".join(__file__.split("/")[:-3])
cfg.config_file = os.path.join(cfg.ROOT, "configs", f"{cfg.config_name}.yaml")
cfg.OUT = os.path.join(*[cfg.ROOT,"experiments",f"{cfg.config_name}"])
cfg.merge_from_file(cfg.config_file)

# cfg.path = os.path.join(cfg.ROOT, "experiments", cfg.config_name)

cfg.world_size = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
cfg.rank = int(os.environ["LOCAL_RANK"]) if "LOCAL_RANK" in os.environ else 0
cfg.world_rank = int(os.environ["RANK"]) if "RANK" in os.environ else 0
cfg.master = cfg.world_rank == 0

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

cfg.distributed = True
# cfg.distributed = cfg.world_size > 1 and cfg.DEVICE != "cpu"

with open(os.environ['PBS_NODEFILE'],'r') as file:
    nodes = [n.strip('\n').split('.')[0] for n in file.readlines()]
cfg.nodename = 'NODE_' + str(nodes.index(socket.gethostname()))

if cfg.LOADER.GPU_BATCH_SIZE is None:
    cfg.LOADER.GPU_BATCH_SIZE = cfg.LOADER.BATCH_SIZE // cfg.world_size

if cfg.master:
    print( f"OMP_NUM_THREADS: {os.environ['OMP_NUM_THREADS']}")
    print("CONFIG:", cfg.config_file, "\n")

time.sleep(cfg.world_rank/4)
print(f'Rank {cfg.world_rank:2d} of {cfg.world_size} online | {cfg.rank} of 4 on {cfg.nodename}')

all_print = False # all nodes print?
if not all_print:
    set_dist_print(cfg.world_rank <= 0)

# TODO: fix cast 1e-3 str to float
cfg.SOLVER.OPTIM.BASE_LR = float(cfg.SOLVER.OPTIM.BASE_LR)
cfg.SOLVER.OPTIM.LR = cfg.SOLVER.OPTIM.BASE_LR * (cfg.world_size ** 0.5) 
cfg.SOLVER.OPTIM.LR = cfg.SOLVER.OPTIM.BASE_LR * (cfg.world_size ** 0.2) 
# cfg.SOLVER.OPTIM.LR = cfg.SOLVER.OPTIM.BASE_LR 

# cfg.freeze() # some of the experiments need it to be mutable

