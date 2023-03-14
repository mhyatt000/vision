from general.config import cfg

class Stopper():
    """Early Stopping"""

    def __init__(self, cond=min, wait=250):
        self.hist = []

        self.wait = wait
        self.waited = 0

    def __call__(self,x):
        """docstring"""

        self.hist.append(x)
        self.best = cond(self.hist)

        if x == self.best:
            self.waited = 0
            return False
        else:
            if self.waited > self.wait:
                return True
            else:
                self.wait += 1

