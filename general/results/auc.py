
from .plotter import Plotter

class AUCPlotter(Plotter):

    def calc(self, points):
        points = sorted(points, key=lambda p: p[0])  # sort points by x value
        x1, y1 = points[0]
        auc = 0.0

        for x2, y2 in points[1:]:
            score = self.trapezoid_score(x1, y1, x2, y2)
            auc += score
            x1, y1 = x2, y2

        return auc


    def show(self, Y, Yh, *args, logits, **kwargs):
        fig, ax = plt.subplots(figsize=(10, 10))
        logits = F.softmax(logits, dim=-1)
        Y = F.one_hot(Y.view(-1).long())

        for i in range(1, cfg.LOADER.NCLASSES):
            # binary classification AUC
            # select only the rows where Y is 1 in column 0 or i
            rows = (Y[:, 0] == 1) | (Y[:, i] == 1)
            probs = logits[rows, 0].view(-1).numpy()
            ova = Y[rows, 0].view(-1).numpy()  # one-vs-all membership

            probs = logits[:, 0].view(-1).numpy()
            ova = Y[:, 0].view(-1).numpy()  # one vs all membership

            tprs, fprs = [], []
            threshs = [0.05 * x for x in list(range(20))]
            # threshs = list(range(-180,180,10))
            for thresh in threshs:
                npand, npnot = np.logical_and, np.logical_not

                tp = npand(probs >= thresh, ova).sum()
                fp = npand(probs >= thresh, npnot(ova)).sum()

                tn = npand(probs < thresh, npnot(ova)).sum()
                fn = npand(probs < thresh, ova).sum()

                tprs.append((tp / (tp + fn)))
                fprs.append((fp / (fp + tn)))

            points = [(x1, y1) for x1, y1 in zip(fprs, tprs)]
            auc = calculate_auc(points)
            ax.plot(fprs, tprs, label=f"{CLASSES[i]:15s} AUC:{auc:.4f}")

        ax.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        ax.legend()
        mkfig("auc.png")

    @staticmethod
    def trapezoid_score(x1, y1, x2, y2):
        width = abs(x2 - x1)
        height = (y1 + y2) / 2
        return width * height
