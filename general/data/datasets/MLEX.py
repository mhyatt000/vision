from torch.datasets import Dataset

# NOTE: is it worth it to subclass all datasets?
# or just added complexity

class MLEXDataset(Dataset):
    """docstring"""

    def __init__(self, ):

    @classmethod
    def __init_subclass__(cls):
        if not hasattr(cls, 'nclasses') or not hasattr(cls, 'size') or not hasattr(cls, 'C'):
            raise NotImplementedError("Child class must define class attributes A, B, and C.")

