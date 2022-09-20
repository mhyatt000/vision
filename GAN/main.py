from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from torch import nn


def update_D(X, Z, D, G, loss, trainer_D):
    """Update discriminator."""

    batch_size = X.shape[0]
    ones = torch.ones((batch_size,), device=X.device)
    zeros = torch.zeros((batch_size,), device=X.device)
    trainer_D.zero_grad()
    real_Y = D(X)
    fake_X = G(Z)

    # Do not need to compute gradient for `G`, detach it from computing gradients.
    fake_Y = D(fake_X.detach())
    loss_D = (
        loss(real_Y, ones.reshape(real_Y.shape))
        + loss(fake_Y, zeros.reshape(fake_Y.shape))
    ) / 2
    loss_D.backward()
    trainer_D.step()
    return loss_D


def update_G(Z, D, G, loss, trainer_G):
    """Update generator."""

    batch_size = Z.shape[0]
    ones = torch.ones((batch_size,), device=Z.device)
    trainer_G.zero_grad()

    # We could reuse `fake_X` from `update_D` to save computation
    fake_X = G(Z)
    # Recomputing `fake_Y` is needed since `D` is changed
    fake_Y = D(fake_X)
    loss_G = loss(fake_Y, ones.reshape(fake_Y.shape))
    loss_G.backward()
    trainer_G.step()
    return loss_G


def train(D, G, data_iter, num_epochs, lr_D, lr_G, latent_dim, data):
    """training process"""

    loss = nn.BCEWithLogitsLoss(reduction="sum")
    for w in D.parameters():
        nn.init.normal_(w, 0, 0.02)
    for w in G.parameters():
        nn.init.normal_(w, 0, 0.02)
    trainer_D = torch.optim.Adam(D.parameters(), lr=lr_D)
    trainer_G = torch.optim.Adam(G.parameters(), lr=lr_G)
    # animator = d2l.Animator(xlabel='epoch', ylabel='loss', xlim=[1, num_epochs], nrows=2, figsize=(5, 5), legend=['discriminator', 'generator'])
    # animator.fig.subplots_adjust(hspace=0.3)
    for epoch in tqdm(range(num_epochs)):
        # Train one epoch
        # metric = d2l.Accumulator(3)  # loss_D, loss_G, num_examples
        for X in data_iter:
            batch_size = X.shape[0]
            Z = torch.normal(0, 1, size=(batch_size, latent_dim))
            # metric.add(update_D(X, Z, D, G, loss, trainer_D), update_G(Z, D, G, loss, trainer_G), batch_size)
        # Visualize generated examples
        Z = torch.normal(0, 1, size=(100, latent_dim))
        fake_X = G(Z).detach().numpy()
        # animator.axes[1].cla()
        # animator.axes[1].scatter(data[:, 0], data[:, 1])
        # animator.axes[1].scatter(fake_X[:, 0], fake_X[:, 1])
        # animator.axes[1].legend(['real', 'generated'])
        # Show the losses
        # loss_D, loss_G = metric[0]/metric[2], metric[1]/metric[2]
        # animator.add(epoch + 1, (loss_D, loss_G))
    # print(f'loss_D {loss_D:.3f}, loss_G {loss_G:.3f}')


def main():
    """main"""

    X = torch.normal(0.0, 1, (1000, 2))
    A = torch.tensor([[1, 2], [-0.1, 0.5]])
    b = torch.tensor([1, 2])
    data = torch.matmul(X, A) + b

    plt.scatter(data[:100, (0)].detach().numpy(), data[:100, (1)].detach().numpy(), label='real')
    print(f"The covariance matrix is\n{torch.matmul(A.T, A)}")
    # plt.show()

    batch_size = 8

    load_tensor = lambda t, bs: torch.reshape( t[: bs * (len(t) // bs), ...], (bs, (len(t) // bs), *t.shape[1:]))
    data_iter = load_tensor(data, batch_size)
    print(data_iter.shape)

    G = nn.Sequential(nn.Linear(2, 2))
    D = nn.Sequential( nn.Linear(2, 5), nn.Tanh(), nn.Linear(5, 3), nn.Tanh(), nn.Linear(3, 1))

    lr_D, lr_G, latent_dim, num_epochs = 0.05, 0.005, 2, 1
    train( D, G, data_iter, num_epochs, lr_D, lr_G, latent_dim, data[:100].detach().numpy())

    "try to make some samples"

    Z = torch.normal(0, 1, size=(100, latent_dim))
    fake_X = G(Z)

    a, b = zip(*fake_X.tolist())

    plt.scatter(a, b, label='fake')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
