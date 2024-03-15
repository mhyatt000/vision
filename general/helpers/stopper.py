class Stopper:
    """Early Stopping"""

    def __init__(self, wait, cond=min,):
        self.hist = []
        self.cond = cond  # condition
        self.wait = wait if wait > 0 else None
        self.waited = 0

    def __call__(self, x):
        """docstring"""

        self.hist.append(x)
        self.best = self.cond(self.hist)

        if self.wait is None:
            return False

        if x == self.best:
            self.waited = 0
            return False
        else:
            if self.waited >= self.wait:
                print("Early Stopping:")
                print(
                    f"no improvement for {self.wait} steps | total steps: {len(self.hist)}"
                )
                print(f"best: {self.best}")
                return True
            else:
                self.waited += 1

    def get_patience(self):
        if self.wait is None:
            return -1
        return self.wait - self.waited

    def get_best(self):
        try:
            return self.best
        except:
            return -1.0
