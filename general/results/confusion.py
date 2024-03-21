import torch
from .plotter import Plotter


class ConfusionPlotter(Plotter):

    def calc(self, Y, Yh):
        if self.cfg.loss.body == "ce":
            self.confusion, self.acc = self.calc_confusion(Y, Yh)
        else:
            raise NotImplementedError
            self.confusion, self.acc = arc_confusion(Y, Yh, centers)

    def show(self, **kwargs):
        """builds confusion matrix"""

        fname = "confusion.png"
        plt.matshow(self.confusion, cmap=plt.cm.Blues, alpha=0.3)
        self.label_matx()

        getname = (
            lambda x: CLASSES[x] if x < len(CLASSES) else f"unknown{x-len(CLASSES)}"
        )
        for i in range(self.confusion.shape[0]):
            for j in range(self.confusion.shape[1]):
                plt.text(
                    x=j,
                    y=i,
                    s=int(self.confusion[i, j]),
                    va="center",
                    ha="center",
                    size="xx-large",
                )

        # plt.title(f"Confusion Matrix")
        plt.xlabel("Predictions")
        plt.ylabel("Ground Truth")
        plt.xticks(
            [i for i in range(len(self.confusion))],
            [getname(i) for i in range(len(self.confusion))],
        )
        plt.yticks(
            [i for i in range(len(self.confusion))],
            [getname(i) for i in range(len(self.confusion))],
        )
        plt.setp(plt.xticks()[1], rotation=30)
        plt.setp(plt.yticks()[1], rotation=30)
        plt.tight_layout()
        self.mkfig(fname, legend=False)

        # for thresh in [55,60,65,70,75]:
        # confusion, acc =  arc_confusion_openset(Y, Yh, centers,thresh)
        # _plot_confusion(confusion,acc,f'confusion_openset{thresh}.png')

    def calc_confusion(self, Y, Yh):
        """calculate confusion matrix"""

        confusion = torch.zeros((self.cfg.loader.nclasses, self.cfg.loader.nclasses))
        Y, Yh = torch.argmax(Y, 1), torch.argmax(Yh, 1)
        for y, yh in zip(Y.view(-1), Yh.view(-1)):
            confusion[y, yh] += 1

        acc = confusion.diag().sum() / confusion.sum(1).sum()
        self.serialize("confusion_from_cross_entropy", confusion)
        return confusion, acc

    def arc_confusion(Y, Yh, centers):
        """confusion matrix with arc embeddings"""

        norm = torch.linalg.norm
        to_rad = lambda a, b: torch.acos(torch.dot(a, b) / (norm(a) * norm(b)))
        angle = lambda a, b: (to_rad(a, b) * 180 / 3.141592)

        Yh = torch.Tensor([[angle(yh, c) for c in centers] for yh in Yh])
        # TODO: if dist is over threshold then put in other category
        # # Y = torch.argmax(Y, 1) # not needed to argmax em
        Yh = torch.argmin(Yh, 1)

        confusion = torch.zeros((self.cfg.loader.nclasses, self.cfg.loader.nclasses))
        for y, yh in zip(Y.cpu().view(-1), Yh.cpu().view(-1)):
            confusion[int(y.item()), int(yh.item())] += 1

        acc = confusion.diag().sum() / confusion.sum(1).sum()
        serialize("confusion_from_centers", confusion)
        return confusion, acc

    def arc_confusion_openset(Y, Yh, centers, thresh):
        """confusion matrix with arc embeddings"""

        norm = torch.linalg.norm
        to_rad = lambda a, b: torch.acos(torch.dot(a, b) / (norm(a) * norm(b)))
        angle = lambda a, b: (to_rad(a, b) * 180 / 3.141592)

        classes = [[] for i in range(len(centers))]
        embeds = [[] for i in range(len(centers))]
        nknown = self.cfg.loader.nclasses

        Y = Y.cpu().view(-1).tolist()
        Yh = Yh.cpu().tolist()

        while Y:
            (y, yh) = (Y.pop(), torch.Tensor(Yh.pop()))
            a = torch.Tensor(
                [angle(yh, c) if c.sum() else 360 for c in centers]
            )  # dont want 0 centers with no values
            i = torch.argmin(a)

            # give priority to known classes
            if any([x < thresh for x in a[:nknown]]):
                classes[i].append(y)
                embeds[i].append(yh)

            # then to unknown classes
            elif any([x < thresh for x in a]):
                classes[i].append(y)
                embeds[i].append(yh)
                # recompute center for that cls
                centers[i] = calc_centers(
                    torch.Tensor([0 for _ in embeds[i]]), torch.stack(embeds[i])
                )[
                    0
                ]  # reuse old code

            # lastly to potential new classes
            else:
                classes.append([y])
                embeds.append([yh])
                centers.append(yh)

        confusion = torch.zeros((len(classes), len(classes)))
        for yh, Y in enumerate(classes):
            for y in Y:
                confusion[int(y), int(yh)] += 1

        acc = confusion.diag().sum() / confusion.sum(1).sum()
        serialize("confusion_from_centers", confusion)
        return confusion, acc

    def show_RKNN_confusion(Y, Yh, rknns, logits, **kwargs):
        """docstring"""

        accs = []
        fprs = []  # false pos rate
        for r, rknn in rknns.items():
            pred = torch.Tensor(rknn.predict(Yh))
            ncol = max([int(x) for x in pred] + [int(x) for x in Y]) + 1
            confusion = torch.zeros((ncol, ncol))

            for y, yh in zip(Y, pred):
                confusion[int(y[0]), int(yh)] += 1

            total = confusion.sum(1).sum()
            tp = confusion.diag().sum()
            acc = tp / total
            accs.append(acc)
            fpr = (total - tp) / total
            fprs.append(fpr)

            _plot_confusion(confusion, acc, f"rknn_openset{r}.png")

        # plot_auc(fprs, accs)
