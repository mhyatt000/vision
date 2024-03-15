from .base import Trainer  # BaseTrainer
# from .cyclegan import CycleGanTrainer

trainers = {
    "BASE": Trainer,
    # "CYCLEGAN": CycleGanTrainer,
}


def _Trainer():
    """trainer factory"""

    return trainers[cfg.trainer]
