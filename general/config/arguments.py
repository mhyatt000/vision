import argparse
import os

from general.utils.dist import set_dist_print
from .defaults import _C as cfg

parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")

parser.add_argument(
    "--config-file",
    default="",
    metavar="FILE",
    help="path to config file",
    type=str,
)
parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument(
    "--skip-test",
    dest="skip_test",
    help="Do not test the final model",
    action="store_true",
)

parser.add_argument(
    "--use-tensorboard",
    dest="use_tensorboard",
    help="Use tensorboardX logger (Requires tensorboardX installed)",
    action="store_true",
    default=False,
)

parser.add_argument(
    "opts",
    help="Modify config options using the command-line",
    default=None,
    nargs=argparse.REMAINDER,
)

parser.add_argument("--save_original_config", action="store_true")
parser.add_argument("--disable_output_distributed", action="store_true")
parser.add_argument("--override_output_dir", default=None)

args = parser.parse_args()

""" done w args """
for k,v in args.__dict__.items():
    setattr(cfg,k,v)

cfg.num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
cfg.distributed = cfg.num_gpus > 1

if cfg.disable_output_distributed:
    set_dist_print(cfg.local_rank <= 0)

file = input('config file: ')+'.yaml'
cfg.config_file = os.path.join('/'.join(__file__.split('/')[:-3]), 'configs',file)
print(cfg.config_file, '\n')

if cfg.config_file:
    cfg.merge_from_file(cfg.config_file)
cfg.merge_from_list(cfg.opts)

""" redundant but ok """
# specify output dir for models
if cfg.override_output_dir:
    cfg.OUTPUT_DIR = cfg.override_output_dir

cfg.freeze()
