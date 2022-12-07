import argparse
import os
import sys

from general.utils.dist import set_dist_print
from .defaults import _C as cfg

parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")

parser.add_argument(
    "--config-name", default="", metavar="FILE", help="path to config file", type=str
)

# parser.add_argument( "--config-file", default="", metavar="FILE", help="path to config file", type=str)
# parser.add_argument("--local_rank", type=int, default=0)
# parser.add_argument( "--skip-test", dest="skip_test", help="Do not test the final model", action="store_true",)
# parser.add_argument( "--use-tensorboard", dest="use_tensorboard", help="Use tensorboardX logger (Requires tensorboardX installed)", action="store_true", default=False,)
# parser.add_argument( "opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER,)
# parser.add_argument("--save_original_config", action="store_true")
# parser.add_argument("--disable_output_distributed", action="store_true")
# parser.add_argument("--override_output_dir", default=None)

args = parser.parse_args()
cfg.config_name = args.config_name
# for k,v in args.__dict__.items():
# setattr(cfg,k,v)

file = (cfg.config_name or input("config file: ")) + ".yaml"

cfg.ROOT = "/".join(__file__.split("/")[:-3])
cfg.config_file = os.path.join(cfg.ROOT, "configs", file)

if cfg.config_file:
    cfg.merge_from_file(cfg.config_file)

cfg.world_size = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
cfg.rank = int(os.environ["RANK"]) if "RANK" in os.environ else 0
cfg.distributed = cfg.world_size > 1 and cfg.DEVICE != "cpu"

dist_print = False
if not dist_print:
    set_dist_print(cfg.rank <= 0)

cfg.freeze()
print('CONFIG:', cfg.config_file, "\n")
