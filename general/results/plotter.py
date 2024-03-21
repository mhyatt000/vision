import json
import os
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt

from general.results import out


class Plotter:

    def __init__(self, cfg, classes=None):
        self.cfg = cfg
        self.classes = classes if classes is not None else [f"c{i}" for i in range(5)]

    def label_matrix(self):
        """Set labels for plt matrix. To be implemented."""
        pass

    @staticmethod
    def is_json_serializable(v):
        try:
            json.dumps(v)
            return True
        except:
            return False

    @classmethod
    def to_json(cls, obj):
        """Convert obj to a version which can be serialized with JSON."""
        if cls.is_json_serializable(obj):
            return obj
        if isinstance(obj, (torch.Tensor, np.ndarray)):
            return obj.tolist()
        if isinstance(obj, dict):
            return {cls.to_json(k): cls.to_json(v) for k, v in obj.items()}
        if isinstance(obj, tuple):
            return tuple(cls.to_json(x) for x in obj)
        if isinstance(obj, list):
            return [cls.to_json(x) for x in obj]
        if hasattr(obj, "__name__") and "lambda" not in obj.__name__:
            return cls.to_json(obj.__name__)
        if hasattr(obj, "__dict__") and obj.__dict__:
            obj_dict = {cls.to_json(k): cls.to_json(v) for k, v in obj.__dict__.items()}
            return {str(obj): obj_dict}
        return str(obj)

    def mkfig(self, fname, legend=None, verbose=True):
        """Save the current figure with optional legend and verbosity."""
        if legend:
            plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out.get_path(self.cfg), fname))
        if verbose:
            print(f"Saved: {fname}")
        plt.close("all")

    def serialize(self, k, v):
        """Serialize and save a key-value pair as JSON."""
        v = self.to_json(v)
        fname = os.path.join(out.get_path(self.cfg), "results.json")
        try:
            with open(fname, "r") as file:
                data = json.load(file)
        except FileNotFoundError:
            data = {}

        data[k] = v
        with open(fname, "w") as file:
            json.dump(data, file, indent=4)

    @abstractmethod
    def calc(self, *args, **kwargs):
        """
        Calculate necessary statistics or results from data.

        Parameters:
            data: The data to calculate results from.
        """
        pass

    @abstractmethod
    def show(self, *args, **kwargs):
        """
        Display the plot or results.
        """
        pass

    def __call__(self, *args, **kwargs):
        """
        Call the calc and show methods in sequence.
        """
        self.calc(*args, **kwargs)
        self.show(*args, **kwargs)
