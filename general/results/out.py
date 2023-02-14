from general.config import cfg
import os

def get_path():
    """dynamically generate output folder"""

    LO = f"{'LO'+cfg.LOADER.LEAVE_OUT if cfg.LOADER.LEAVE_OUT is not None else ''}"
    fold = f"{cfg.SEED}.{cfg.SWAP}" if cfg.EXP == '5x2' else ''
    version = LO + fold

    out = [
        cfg.ROOT,
        "experiments",
        f"{cfg.config_name}",
    ]
    out = [o for o in out if o]
    print(out)
    return os.path.join(*out)
