from general.config import cfg


def get_path():
    """dynamically generate output folder"""

    out = [
        cfg.ROOT,
        "experiments",
        f"{cfg.config_name}",
        f"{'LO'+cfg.LOADER.LEAVE_OUT if cfg.LOADER.LEAVE_OUT is not None else ''}_{cfg.SEED}.{swap if 'swap' in cfg else ''}",
    ]
    out = [o for o in out if o]
    print(out)
    return os.path.join(out)
