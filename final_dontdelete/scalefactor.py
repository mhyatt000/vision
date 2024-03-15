import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

fig, ax = plt.subplots()
for s in [1, 64]:

    losses = []
    P = []
    vals = []

    steps = torch.linspace(-10, 10, steps=100).tolist()
    for val in steps:

        vals.append(val)
        a = torch.Tensor([[val, 10, 1]])
        a *= s
        b = torch.Tensor([[1, 0, 0]])

        a = F.softmax(a)

        p = a[0][0].item()
        P.append(p)

        loss = F.cross_entropy(a, b)
        losses.append(loss)
        print(p)
        print(loss)
        print()

    ax.scatter(P, losses, label=f"s: {s}")
    # ax.scatter(vals, losses, label=f"s: {s}")

ax.set(
    title="Effect of scale factor s on loss severity",
    ylabel='Loss',
    xlabel='probability of yhat = y',
    # xlabel='logit value before scaled (prob={0..1})',
)
plt.legend()
plt.show()
