import os
from yacs.config import CfgNode
import yaml


class CN(CfgNode):
    """docstring"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def default(self, file=""):
        """init from default in the same dir"""

        path = "/".join(__file__.split("/")[:-1]) + "/" + (file or "default.yaml")
        with open(path, "r") as file:
            data = yaml.safe_load(file)
        return self.build(data)

    def build(self, data):
        """builds a node"""

        for k, v in data.items():
            if type(v) not in [list, dict, tuple]:
                setattr(self, k, v)
            else:
                setattr(self, k, CN().build(v))

        return self


_C = CN().default()
print(_C)
