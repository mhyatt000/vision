import argparse
import time
import os
import sys

from general.utils.dist import set_dist_print
from .defaults import _C as cfg


parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")

parser.add_argument( "--config-name", default="", metavar="FILE", help="path to config file", type=str)

# parser.add_argument( "--config-file", default="", metavar="FILE", help="path to config file", type=str)
# parser.add_argument("--local_rank", type=int, default=0)
# parser.add_argument( "--skip-test", dest="skip_test", help="Do not test the final model", action="store_true",)
# parser.add_argument( "--use-tensorboard", dest="use_tensorboard", help="Use tensorboardX logger (Requires tensorboardX installed)", action="store_true", default=False,)
# parser.add_argument( "opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER,)
# parser.add_argument("--save_original_config", action="store_true")
# parser.add_argument("--disable_output_distributed", action="store_true")
# parser.add_argument("--override_output_dir", default=None)

args = parser.parse_args()
cfg.config_name = args.config_name.split('/')[-1].replace('.yaml','')
# for k,v in args.__dict__.items():
# setattr(cfg,k,v)

cfg.config_name = cfg.config_name or input("config file: ")
cfg.ROOT = "/".join(__file__.split("/")[:-3])
cfg.config_file = os.path.join(cfg.ROOT, "configs", f"{cfg.config_name}.yaml")
cfg.merge_from_file(cfg.config_file)

# cfg.path = os.path.join(cfg.ROOT, "experiments", cfg.config_name)

cfg.world_size = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
cfg.rank = int(os.environ["LOCAL_RANK"]) if "LOCAL_RANK" in os.environ else 0
cfg.world_rank = int(os.environ["RANK"]) if "RANK" in os.environ else 0

if False:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD

    cfg.world_size = int(comm.Get_size())
    cfg.world_rank = int(comm.Get_rank())

    os.environ['RANK'] = str(cfg.world_rank)
    os.environ['WORLD_SIZE'] = str(cfg.world_size)
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.rank)

    with open(os.environ['PBS_NODEFILE'],'r') as file:
        nodes = [n.strip('\n') for n in file.readlines()]

    nodenames = {n.split('.')[0]:'NODE_'+str(i) for i,n in enumerate(nodes)}
    ppn = cfg.world_size // len(nodes) # process per node

    cfg.nodename = nodenames[MPI.Get_processor_name()]
    cfg.local_rank = cfg.world_rank % ppn

    os.environ['MASTER_ADDR'] = nodes[0]
    os.environ['MASTER_PORT'] = str(2345)

if True:
    print('no OMPI')

cfg.distributed = cfg.world_size > 1 and cfg.DEVICE != "cpu"

if cfg.LOADER.GPU_BATCH_SIZE is None:
    cfg.LOADER.GPU_BATCH_SIZE = cfg.LOADER.BATCH_SIZE // cfg.world_size

if not cfg.world_rank:
    print(
        f"OMP_NUM_THREADS: {os.environ['OMP_NUM_THREADS'] if 'OMP_NUM_THREADS' in os.environ else -1}"
    )
    print("CONFIG:", cfg.config_file, "\n")

time.sleep(cfg.world_rank/2)
# print(f"Rank: {cfg.world_rank} online")
try: 
    print(f'Rank {cfg.world_rank} of {cfg.world_size} online | {cfg.local_rank} of {ppn} on {cfg.nodename}')
except:
    print(f'Rank {cfg.world_rank} of {cfg.world_size} online | {cfg.rank} of NODE_0 on localhost')

dist_print = False
if not dist_print:
    set_dist_print(cfg.world_rank <= 0)

cfg.SOLVER.OPTIM.LR = cfg.SOLVER.OPTIM.BASE_LR * cfg.world_size

# cfg.freeze() # some of the experiments need it to be mutable
