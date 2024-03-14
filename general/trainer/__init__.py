from .base import BaseTrainer
from .cyclegan import CycleGanTrainer

trainers = {
        "BASE":BaseTrainer,
        "CYCLEGAN":CycleGanTrainer,
    }

def _Trainer():
    """trainer factory"""

    return trainers[cfg.trainer]
