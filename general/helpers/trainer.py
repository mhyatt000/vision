import os

import matplotlib.pyplot as plt
from sklearn.decomposition import KernelPCA as KPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE
# ...
import torch
from torch import distributed as dist
from torch import optim
from torch.cuda.amp.grad_scaler import GradScaler
from tqdm import tqdm

from general.config import cfg
from general.data import build_loaders
from general.helpers import Checkpointer, make_optimizer, make_scheduler
from general.losses import make_loss

from torch.cuda.amp import  autocast
from torch.cuda.amp import GradScaler

# import plotly.express as px
# from general.solver import make_lr_scheduler, make_optimizer

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
        self.criterion = make_loss()

        params = [{"params":model.parameters()}]
        if cfg.LOSS.BODY == 'AAM':
            params.append({"params": self.criterion.parameters()})

        self.optimizer = make_optimizer(params)
        self.scaler = GradScaler(growth_interval=100) # default is 2k
        self.scheduler = make_scheduler(self.optimizer)
        self.ckp = Checkpointer(self)
        self.loaders = build_loaders()
        self.scaler = GradScaler()

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
        self.best_epoch = 0

        self.use_decay = cfg.SOLVER.WEIGHT_DECAY_SCHEDULE
        self.milestone_tgt = 0

        # state
        self.epoch, self.batch = 0, 0
        self.losses, self.accs = [], [0]
        self.best_epoch = 0

        # try to load from snapshot ... must be last
        self.ckp.load()
        print()

    def update(self):
        """update after the training loop"""

    def patient(self):
        """given the eval result should we terminate the loop"""

        if cfg.LOSS.BODY == "AAM":
            self.ckp.save()
            self.best_eopch = self.epoch
            return

        if self.accs[-1] < self.best:
            self.patience += 1
        else:
            self.patience = 0
            self.best = self.accs[-1]
            self.best_epoch = self.epoch
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

        self.optimizer.zero_grad()

        with autocast(cfg.AMP):
            X, Y = X.to(cfg.DEVICE), Y.to(cfg.DEVICE)
            Yh = self.model(X).view((Y.shape[0], -1))

            # backwards pass
            loss = self.criterion(Yh, Y)

        #TODO: make robust ... if amp then use amp else regular
        self.scaler.scale(loss.to(cfg.DEVICE)).backward() # loss.backward()

        #TODO: can you make this into a hook?
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.SOLVER.GRAD_CLIP)

        self.scaler.step(self.optimizer) # self.optimizer.step()
        self.scaler.update()

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
                desc = f'{self.epoch}/{cfg.SOLVER.MAX_EPOCH} | loss: {"%.4f" % self.loss} | amp: {"%4f" % self.scaler.get_scale()} '
                t.set_description(desc)  # acc: {"%.4f" % acc}')
                t.update()

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
            self.eval()

            self.epoch += 1
            self.batch = 0

            # save model and state
            self.ckp.epoch = self.epoch
            self.patient()

            if t:
                t.set_description(f"Epochs | best: {'%.4f' % self.best} @ {self.best_epoch} | lr: {'%.4f' % self.scheduler.get_last_lr()[0]} ")
                t.update()


    def run(self):
        """trains model"""

        if cfg.TRAINER.TRAIN:
            self.train()
        if cfg.TRAINER.VAL:
            self.eval()
            # self.summary()

    def eval(self):
        """docstring"""

        if not cfg.TRAINER.VAL:
            return
        if cfg.LOSS.BODY == "AAM":
            if cfg.TRAINER.TRAIN:
                return
            Y, Yh = self.embed()
            self.population_distance(Y, Yh)
            self.show_embeddings(Y, Yh)
        if cfg.LOSS.BODY == "CE":
            self.classwise_accuracy()


    def embed(self):
        """docstring"""

        loader = self.loaders["test"]
        allY, allYh = [], []
        t = tqdm(total=len(loader), leave=False) if not (cfg.rank and cfg.distributed) else None
        with torch.no_grad():
            for X, Y in loader:

                X, Y = X.to(cfg.DEVICE), Y.to(cfg.DEVICE)
                Yh = self.model(X).view((Y.shape[0], -1))
                Y, Yh = gather(Y), gather(Yh)

                allY.append(Y)
                allYh.append(Yh)

                if t:
                    t.set_description(f"Embed")
                    t.update()

        return torch.cat(allY), torch.cat(allYh)


    def classwise_accuracy(self):
        """get classwise accuracy"""

        Y, Yh = self.embed()

        """
        Yh = self.fc(Yh)
        if cfg.LOSS.BODY != "CE":
            Y = torch.Tensor([[int(i == y) for i in range(5)] for y in Y]).to(cfg.DEVICE)
        """

        confusion = torch.zeros((cfg.LOADER.NCLASSES, cfg.LOADER.NCLASSES))
        Y, Yh = torch.argmax(Y, 1), torch.argmax(Yh, 1)
        for y, yh in zip(Y.view(-1), Yh.view(-1)):
            confusion[y, yh] += 1

        acc = confusion.diag().sum() / confusion.sum(1).sum()

        # print(confusion.tolist())
        self.accs.append(acc)
        self.confusion = confusion
        # print(confusion.diag() / confusion.sum(1))

        self.show_confusion()
        self.show_acc()

    def show_confusion(self):
        """builds confusion matrix"""

        plt.rcParams.update({"font.size": 18})
        fig, ax = plt.subplots(figsize=(7.5, 7.5))
        ax.matshow(self.confusion, cmap=plt.cm.Blues, alpha=0.3)

        for i in range(self.confusion.shape[0]):
            for j in range(self.confusion.shape[1]):
                ax.text(
                    x=j, y=i, s=int(self.confusion[i, j]), va="center", ha="center", size="xx-large"
                )

        ax.set(
            xlabel="Predictions",
            ylabel="Ground Truth",
            title=f"Confusion Matrix | {cfg.config_name} {self.epoch}/{cfg.SOLVER.MAX_EPOCH}",
        )
        plt.savefig(os.path.join(self.ckp.path, "confusion.png"))
        plt.close()


    def population_distance(self, Y, Yh):
        """measures if positive pairs are different from negative pairs"""

        phist = {i: [] for i in range(cfg.LOADER.NCLASSES)}
        nhist = {i: [] for i in range(cfg.LOADER.NCLASSES)}
        C = set(range(cfg.LOADER.NCLASSES))

        Y = Y.view(-1)

        for c in C:
            pos = Yh[(Y == c).view(-1)]
            neg = Yh[(Y != c).view(-1)]

            dist = lambda a, b: (a - b).pow(2).sum(-1).sqrt()
            angle = ( lambda a, b: torch.acos( torch.dot(a, b) / (torch.linalg.norm(a) * torch.linalg.norm(b))) * 180 / 3.141592)

            n = len(pos) // 2
            ppairs = [angle(i, j) for i, j in zip(pos[:n], pos[n : 2 * n])]
            npairs = [angle(i, j) for i, j in zip(pos[:n], neg[n : 2 * n])]

            phist[c] = phist[c] + ppairs
            nhist[c] = nhist[c] + npairs

        pall, nall = [], []
        for c in C:
            pall += [float(x) if not torch.isnan(x) else -1 for x in phist[c][:1000]]
            nall += [float(x) for x in nhist[c][:1000]]

        fig, ax = plt.subplots()
        ax.hist(pall, label='positive', bins=30, alpha=0.5)
        ax.hist(nall, label='negative', bins=30, alpha=0.5)
        ax.set(title="Population Distance (d`)", xlabel="angle", ylabel="frequency")
        plt.legend()
        plt.savefig(os.path.join(self.ckp.path, "dist.png"))
        plt.close()

    def show_acc(self):
        """shows accuracy / time in a png"""

        fig, ax = plt.subplots()
        ax.plot([i for i in range(len(self.accs))], self.accs)
        ax.set(title="accuracy / time", xlabel="epochs", ylabel="accuracy")
        plt.savefig(os.path.join(self.ckp.path,"acc.png"))
        plt.close()

    def show_loss(self):
        """shows loss / time in a png"""

        fig, ax = plt.subplots()
        ax.plot([i for i in range(len(self.losses))], self.losses)
        ax.set(title="loss / time", xlabel="epochs", ylabel="loss")
        plt.savefig(os.path.join(self.ckp.path,"loss.png"))
        plt.close()

    def show_embeddings(self, Y, Yh):
        """docstring"""

        lda = LDA(n_components=2,solver='svd')
        # tsne = TSNE(n_components=3, random_state=cfg.SOLVER.SEED)
        # kpca = KPCA(n_components=2, kernel='poly', gamma=15, random_state=cfg.SOLVER.SEED)

        Yh = lda.fit_transform(Yh.cpu().numpy(),Y.cpu().numpy())

        fig, ax = plt.subplots()
        # ax = fig.add_subplot(projection="3d")

        scatter = ax.scatter(Yh[:,0], Yh[:,1], c=Y.view(-1).tolist())
        # ax.scatter(Yh[:,0], Yh[:,1],Yh[:,2], c=Y.view(-1).tolist())
        # ax.view_init(0, 180)

        ax.legend(*scatter.legend_elements())
        plt.savefig(os.path.join(self.ckp.path,"embed.png"))
        plt.close()

    def summary(self):
        """gives a summary of embeddings"""

        if cfg.LOSS.BODY != "AAM":
            return  # only for embeddings

        Y, Yh = self.embed()

        scale = lambda X: torch.div( X, torch.sqrt(torch.sum(torch.pow(X, 2), -1)).reshape(-1, 1))

        # find center vector
        for c in set([i[0] for i in tgts.tolist()]):

            embed = Yh[(Y == c).view(-1)]
            embed = scale(embed)
            x = scale(torch.mean(embed, -2))
            loss = torch.nn.CrossEntropyLoss()(embed, x.repeat(embed.shape[0], 1))
            print(f"cls {int(c)} | nsamples: {embed.shape[0]} | loss: {float(loss)}")
