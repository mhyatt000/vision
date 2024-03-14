class Animator():
    """custom animator"""

    def __init__(self, ):
        pass



        animator = d2l.Animator(
            xlabel="epoch",
            ylabel="loss",
            xlim=[1, num_epochs],
            nrows=2,
            figsize=(5, 5),
            legend=["discriminator", "generator"],
        )
        animator.fig.subplots_adjust(hspace=0.3)

        metric.add(
            d2l.update_D(X, Z, net_D, net_G, loss, trainer_D),
            d2l.update_G(Z, net_D, net_G, loss, trainer_G),
            batch_size,
        )



class Trainer():
    """docstring"""

    def __init__(self, ):
        pass
