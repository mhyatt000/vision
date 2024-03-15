from tqdm import tqdm


def prog(cfg, length, desc=None):
    """
    creates a tqdm progress bar for the wrapped function when it is called
    of length length and desc equal to function output or desc
    only if there is not one already
    only if this is a node that can print to stdout
    iterates the bar on each call ... nice for loops
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            if not hasattr(wrapper, "tqdm"):
                wrapper.tqdm = tqdm(total=length, desc=desc)  # , leave=False)
            result = func(*args, **kwargs)
            if result:
                wrapper.tqdm.set_description(result)
            wrapper.tqdm.update()

            return result

        return wrapper if cfg.master else func  # no tqdm if not master

    return decorator
