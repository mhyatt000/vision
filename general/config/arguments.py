from argparse import ArgumentParser as AP
import json
import socket
import torch

import time
import os
import sys

from general.utils.dist import set_dist_print
from .defaults import _C as cfg

ap = AP(description="PyTorch Object Detection Training")

ap.add_argument( "--config-file", default="",  help="path to config file", type=str)

args = ap.parse_args()
cfg.ROOT = "/".join(__file__.split("/")[:-3])

# strip so it can be used for experiment folder
file = args.config_file.split('/')
print(file)
file = file[file.index("configs"):-1] + [file[-1].replace('.yaml','') ]

print(file)
quit()

cfg.config_file = os.path.join(cfg.ROOT, "configs", f"{file}.yaml")
cfg.OUT = os.path.join(cfg.ROOT,"experiments",f"{file}")
cfg.merge_from_file(cfg.config_file)

def import_cfg():
    """TODO
    if IMPORT is specified in a cfg file,
    recursively override default with child and child with parent
    """
    pass


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

try:
    with open(os.environ['PBS_NODEFILE'],'r') as file:
        cfg.nodes = [n.strip('\n').split('.')[0] for n in file.readlines()]
    cfg.nodename = socket.gethostname()
    cfg.nodenumber = 'NODE_' + str(cfg.nodes.index(cfg.nodename))
except: # probably using the login node :(
    pass
    cfg.nodes = ['LOGIN']
    cfg.nodename = cfg.nodes[0]
    cfg.nodenumber = 0

if cfg.LOADER.GPU_BATCH_SIZE is None:
    cfg.LOADER.GPU_BATCH_SIZE = cfg.LOADER.BATCH_SIZE // cfg.world_size

all_print = False # all nodes print?
if not all_print:
    set_dist_print(cfg.world_rank <= 0)

# TODO: fix cast 1e-3 str to float
cfg.SOLVER.OPTIM.BASE_LR = float(cfg.SOLVER.OPTIM.BASE_LR)
# cfg.SOLVER.OPTIM.LR = cfg.SOLVER.OPTIM.BASE_LR * (cfg.world_size ** 0.5) 
# cfg.SOLVER.OPTIM.LR = cfg.SOLVER.OPTIM.BASE_LR * (cfg.world_size ** 0.2) 
cfg.SOLVER.OPTIM.LR = cfg.SOLVER.OPTIM.BASE_LR 

# cfg.freeze() # some of the experiments need it to be mutable

