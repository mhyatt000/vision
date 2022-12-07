import os

import matplotlib.pyplot as plt
import torch
from torch import distributed as dist
from torch import optim
from torch.cuda.amp.grad_scaler import GradScaler
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau, StepLR
from tqdm import tqdm

from general.config import cfg
from general.data import build_loaders
from general.helpers import Checkpointer
from general.losses import make_loss

# import plotly.express as px
# from general.solver import make_lr_scheduler, make_optimizer


OPTIM = {
    "ADAM": optim.Adam,
    "SGD": optim.SGD,
}


def make_optimizer(model):
    return OPTIM[cfg.OPTIM.BODY](
        model.parameters(),
        lr=cfg.OPTIM.LR,
        betas=cfg.OPTIM.BETAS,
        weight_decay=float(cfg.OPTIM.DECAY),
    )


def make_scheduler(optimizer):
    return StepLR(optimizer, step_size=1, gamma=0.5)


def gather(x):
    """simple all gather manuver"""

    if not cfg.distributed:
        return x

    _gather = [torch.zeros(x.shape, device=cfg.DEVICE) for _ in range(dist.get_world_size())]
    dist.all_gather(_gather, x)
    return torch.cat(_gather)


class Trainer:
    """manages and abstracts options from the training loop"""

    def __init__(self, model):

        # essentials
        self.model = model
        self.optimizer = make_optimizer(model)
        self.scaler = GradScaler()
        self.scheduler = make_scheduler(self.optimizer)
        self.criterion = make_loss()
        self.ckp = Checkpointer(self)
        self.loaders = build_loaders()

        # for validation
        """
        self.fc = torch.nn.Linear(cfg.LOSS.PFC.EMBED_DIM, cfg.LOSS.PFC.NCLASSES)
        self.fc = self.fc.to(cfg.DEVICE)
        self.val_optim = make_optimizer(self.fc)
        self.val_crit = torch.nn.CrossEntropyLoss()
        """

        """what is ema"""

        # early stopping
        self.use_patience = cfg.SOLVER.AUTO_TERMINATE_PATIENCE != -1
        self.patience = 0
        self.max_patience = cfg.SOLVER.AUTO_TERMINATE_PATIENCE
        self.best = 0.0

        self.use_decay = cfg.SOLVER.WEIGHT_DECAY_SCHEDULE
        self.milestone_tgt = 0

        # state
        self.epoch, self.batch = 0, 0
        self.losses, self.accs = [], []

        # try to load from snapshot ... must be last
        self.ckp.load()
        print()

    def update(self):
        """update after the training loop"""

    def patient(self):
        """given the eval result should we terminate the loop"""

        if self.accs[-1] < self.best:
            self.patience += 1
        else:
            self.patience = 0
            self.best = self.accs[-1]
            self.ckp.save()

        if self.use_patience and self.patience >= self.max_patience:
            print()
            print(f"Auto Termination at {self.epoch}, current best {self.best}")
            quit()

    def init_decay():
        """docstring"""

        # Adapt the weight decay
        if cfg.SOLVER.WEIGHT_DECAY_SCHEDULE and hasattr(scheduler, "milestones"):
            milestone_target = 0
            for i, milstone in enumerate(list(scheduler.milestones)):
                if scheduler.last_epoch >= milstone * cfg.SOLVER.WEIGHT_DECAY_SCHEDULE_RATIO:
                    milestone_target = i + 1

    def iter(self, X, Y):
        """performs a training step after inference"""

        X, Y = X.to(cfg.DEVICE), Y.to(cfg.DEVICE)
        self.optimizer.zero_grad()
        Yh = self.model(X).view((Y.shape[0], -1))

        # print(Yh[0])
        # print(Y[0])

        # backwards pass
        loss = self.criterion(Yh, Y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.SOLVER.GRAD_CLIP)

        # self.scaler.scale(loss).backward()
        # self.scaler.step(self.optimizer)
        # self.scaler.update()

        self.optimizer.step()
        self.loss = float(loss)

        # fitting fc for validation
        """
        if cfg.LOSS.BODY != "CE":
            self.val_optim.zero_grad()
            Yh = self.fc(Yh.detach().clone())
            Y = torch.Tensor([[int(i == y) for i in range(5)] for y in Y]).to(cfg.DEVICE)
            loss = self.val_crit(Yh, Y)
            loss.backward()
            self.val_optim.step()
        """

        """TODO make msg messenger obj to handle reporting
        and documenting (maybe a graph?)
        """

        if self.batch == 0:
            self.losses.append(self.loss)
        # self.mk_figure(Y, Yh)
        self.batch += 1

    def step(self):
        """steps after an epoch"""

        loader = self.loaders["train"]
        if cfg.distributed:
            loader.sampler.set_epoch(self.epoch)

        t = tqdm(total=len(loader), leave=False) if not (cfg.rank and cfg.distributed) else None
        for X, Y in loader:
            self.iter(X, Y)

            if t:
                desc = f'{self.epoch}/{cfg.SOLVER.MAX_EPOCH} | loss: {"%.4f" % self.loss} | '

                t.set_description(desc)  # acc: {"%.4f" % acc}')
                t.update()

        self.epoch += 1
        self.batch = 0
        self.scheduler.step()

    def train(self):
        """trains model"""

        print("begin training loop...")

        t = (
            tqdm(total=cfg.SOLVER.MAX_EPOCH, desc="Epochs", leave=False)
            if not (cfg.rank and cfg.distributed)
            else None
        )
        _ = t.update(self.epoch) if t else None

        for epoch in range(self.epoch, cfg.SOLVER.MAX_EPOCH):

            self.step()
            self.show_loss()
            _ = t.update() if t else None

            self.eval()

            # save model and state
            self.ckp.epoch = self.epoch
            self.patient()

    def run(self):
        """trains model"""

        if cfg.TRAINER.TRAIN:
            self.train()
        if cfg.TRAINER.VAL:
            self.eval()
            self.summary()

    def eval(self):
        """get classwise accuracy"""

        if not cfg.TRAINER.VAL:
            return
        if cfg.LOSS.BODY != "CE":
            return

        loader = self.loaders["test"]

        confusion = torch.zeros(cfg.LOADER.NCLASSES, cfg.LOADER.NCLASSES)
        t = tqdm(total=len(loader), leave=False) if not (cfg.rank and cfg.distributed) else None
        with torch.no_grad():
            for X, Y in loader:

                X, Y = X.to(cfg.DEVICE), Y.to(cfg.DEVICE)
                Yh = self.model(X).view((Y.shape[0], -1))
                Y, Yh = gather(Y), gather(Yh)

                """
                Yh = self.fc(Yh)
                if cfg.LOSS.BODY != "CE":
                    Y = torch.Tensor([[int(i == y) for i in range(5)] for y in Y]).to(cfg.DEVICE)
                """

                _, Yh = torch.max(Yh, 1)
                for y, yh in zip(Y.view(-1), Yh.view(-1)):
                    confusion[y.long(), yh.long()] += 1

                acc = confusion.diag().sum() / confusion.sum(1).sum()
                # acc = -1
                if t:
                    t.set_description(f"validation loop...  acc: {'%.4f' % acc} | ")
                    t.update()

        self.accs.append(acc)
        self.confusion = confusion
        # print(confusion.diag() / confusion.sum(1))
        self.show_confusion()
        self.show_acc()

    def show_confusion(self):
        """builds confusion matrix"""

        fig, ax = plt.subplots(figsize=(7.5, 7.5))
        ax.matshow(self.confusion, cmap=plt.cm.Blues, alpha=0.3)

        for i in range(self.confusion.shape[0]):
            for j in range(self.confusion.shape[1]):
                ax.text(
                    x=j, y=i, s=int(self.confusion[i, j]), va="center", ha="center", size="xx-large"
                )

        plt.rcParams.update({'font.size': 18})
        ax.set(xlabel="Predictions", ylabel="Ground Truth", title="Confusion Matrix")
        plt.savefig(os.path.join(self.ckp.path, "confusion.png"))
        plt.close()

    def show_acc(self):
        """shows accuracy / time in a png"""

        fig, ax = plt.subplots()
        ax.plot([i for i in range(len(self.accs))], self.accs)
        ax.set(title="accuracy / time", xlabel="epochs", ylabel="accuracy")
        plt.savefig("acc.png")
        plt.close()

    def show_loss(self):
        """shows loss / time in a png"""

        fig, ax = plt.subplots()
        ax.plot([i for i in range(len(self.losses))], self.losses)
        ax.set(title="loss / time", xlabel="epochs", ylabel="loss")
        plt.savefig("loss.png")
        plt.close()

    def show_embeddings(self, Y, Yh):
        """docstring"""

        fig, ax = plt.subplots()
        ax = fig.add_subplot(projection="3d")

        for c in set(Y.tolist()):
            yh = Yh[Y == c].tolist()
            ax.scatter(yh[..., 0], yh[..., 1], yh[..., 2])

        plt.savefig("temp.png")
        plt.close()

    def summary(self):
        """gives a summary of embeddings"""

        if cfg.LOSS.BODY != "AAM":
            return  # only for embeddings

        loader = self.loaders["test"]

        """
        DO YOU NEED TO TURN OFF DISTRIBUTED SAMPLING??
        """

        srcs, tgts = [], []
        t = tqdm(total=len(loader), leave=False) if not (cfg.rank and cfg.distributed) else None
        for X, Y in loader:
            with torch.no_grad():

                X, Y = X.to(cfg.DEVICE), Y.to(cfg.DEVICE)
                Yh = self.model(X).view((Y.shape[0], -1))

                srcs.append(Yh)
                tgts.append(Y)

                if t:
                    t.set_description(f"Summary")
                    t.update()

        srcs, tgts = torch.cat(srcs), torch.cat(tgts)
        for c in set([i[0] for i in tgts.tolist()]):

            embed = srcs[(tgts == c).view(-1)]

            # find center vector
            scale = lambda X: torch.div(
                X, torch.sqrt(torch.sum(torch.pow(X, 2), -1)).reshape(-1, 1)
            )
            embed = scale(embed)
            x = scale(torch.mean(embed, -2))

            loss = torch.nn.CrossEntropyLoss()(embed, x.repeat(embed.shape[0], 1))
            print(f"cls {int(c)} | nsamples: {embed.shape[0]} | loss: {float(loss)}")
