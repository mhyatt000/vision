"mostly used in 5x2 experiment rn"

import json
from general.config import cfg
import os
from os.path import join

# string to dict
# _s2d = lambda s: {k:v for k,v in [x.split(':') for x in s.split('_')]}
# s2d = lambda s: {k:int(v) for k,v in _s2d(s).items()}
d2s = lambda x: '_'.join([f'{k}:{v}' for k,v in x.items()])

def get_exp_version():
    """return version string if there is one"""

    if 'versions.json' in os.listdir(cfg.OUT):
        with open(join(cfg.OUT,'versions.json'),'r') as file:
            versions = json.load(file)

        if cfg.EXP.BODY == '5x2' and cfg.EXP.PARTITION:
            versions = [{k:v} for k,v in versions.items() if k == cfg.nodename]
            return versions[0][cfg.nodename]

        return versions[0]

def get_path():
    """dynamically generate output folder"""

    version = get_exp_version()
    return join(cfg.OUT,d2s(version)) if version else cfg.OUT
