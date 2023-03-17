"mostly used in 5x2 experiment rn"

import json
from general.config import cfg
import os
from os.path import join

# string to dict
d2s = lambda x: '_'.join([f'{k}:{v}' for k,v in x.items()])
# _s2d = lambda s: {k:v for k,v in [x.split(':') for x in s.split('_')]}
# s2d = lambda s: {k:int(v) for k,v in _s2d(s).items()}

def get_exp_version():
    """return version string if there is one"""

    if 'experiments.json' in os.listdir(cfg.OUT):
        with open(join(cfg.OUT,'experiments.txt'),'r') as file:
            version = json.load(file)[0]
        return version

def get_path():
    """dynamically generate output folder"""

    version = d2s(get_exp_version())
    return join(cfg.OUT,version) if version else cfg.OUT
