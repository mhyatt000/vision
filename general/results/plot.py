import torch
import matplotlib as plt
from general import config as cfg

def plot_loss(loss, lr=None):
    """plots loss over time"""

    fig, ax = plt.subplots()
    epoch_size = len(loss) // cfg.SOLVER.MAX_EPOCH
    X = [i for i,_ in enumerate(loss)]
    plt.plot(X, loss, label='loss')
    if lr:
        plt.plot(X, lr, label='learning rate')

    epochs = list(range(cfg.SOLVER.MAX_EPOCH))
    plt.xticks([i*epoch_size for i in epochs], epochs)
    plt.legend()
    plt.savefig(os.path.join(self.ckp.path, "loss.png"))
    plt.close('all')
