from torch import optim
from general.config import cfg

OPTIM = {
    "ADAM": optim.Adam,
    "SGD": optim.SGD,
}


def make_optimizer(params):
    #TODO: add support for *args where args are objects to be optimized
    # ie: make_optim(*args, params):
    # assert not (args and params)

    kwargs = dict(
        params=params,
        lr=cfg.OPTIM.LR,
        weight_decay=float(cfg.OPTIM.DECAY),
    )

    if cfg.OPTIM.BODY == "SGD":
        kwargs['momentum']=cfg.OPTIM.MOMENTUM
    if cfg.OPTIM.BODY == 'ADAM':
          kwargs['betas'] = cfg.OPTIM.BETAS

    return OPTIM[cfg.OPTIM.BODY](**kwargs)
