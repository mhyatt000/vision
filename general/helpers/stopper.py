class Stopper:
    """ Early Stopping
    to halt a process based on a specified condition over a sequence of values.
    
    Attributes:
        hist (list): A history of the observed values.
        cond (function): The condition function used to evaluate the best value in the history. Defaults to `min` but can
                         be `max` for cases where improvement is defined as an increase in the monitored value.
        patience (int): The number of steps to wait for an improvement before stopping. If an improvement is observed, the
                    counter resets.
        waited (int): The current number of steps waited since the last improvement.
        best (var): The best value observed according to the condition function.
    
    Parameters:
        patience (int): The patience or number of steps to wait without improvement before stopping.
        cond (function): The function to determine the best value observed. Typically `min` for minimization tasks and
                         `max` for maximization tasks.
    """

    def __init__(self, patience, cond=min):
        self.hist = []
        self.cond = cond  # condition
        self.patience = patience if patience > 0 else None
        self.waited = 0

    def __call__(self, x):
        """docstring"""

        self.hist.append(x)
        self.best = self.cond(self.hist)

        if self.patience is None:
            return False

        if x == self.best:
            self.waited = 0
            return False
        else:
            if self.waited >= self.patience:
                print("Early Stopping:")
                print(
                    f"no improvement for {self.patience} steps | total steps: {len(self.hist)}"
                )
                print(f"best: {self.best}")
                return True
            else:
                self.waited += 1

    def get_patience(self):
        if self.patience is None:
            return -1
        return self.patience - self.waited

    def get_best(self):
        try:
            return self.best
        except:
            return -1.0
