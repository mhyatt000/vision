"mostly used in 5x2 experiment rn"

import json
import os
from os.path import join

# string to dict
# _s2d = lambda s: {k:v for k,v in [x.split(':') for x in s.split('_')]}
# s2d = lambda s: {k:int(v) for k,v in _s2d(s).items()}
d2s = lambda x: '_'.join([f'{k}:{v}' for k,v in x.items()])

def get_exp_version(cfg):
    """return version string if there is one"""

    if 'versions.json' in os.listdir(cfg.exp.out):
        with open(join(cfg.exp.out,'versions.json'),'r') as file:
            versions = json.load(file)

        if cfg.exp.body == 'SERIES' and cfg.exp.partition:
            versions = [{k:v} for k,v in versions.items() if k == cfg.nodename]
            return versions[0][cfg.nodename]

        return versions[0]

def get_path(cfg):
    """dynamically generate output folder"""

    version = get_exp_version(cfg)
    return join(cfg.exp.out,d2s(version)) if version else cfg.exp.out
