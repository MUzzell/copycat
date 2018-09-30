
import numpy as np


class AEGreedy(object):

    def __init__(self, start, end, endt, l_start, *args, **kwargs):
        self.start = start
        self.end = end
        self.endt = endt

        self.l_start = l_start

    def greedy(self, steps):
        start_diff = self.start - self.end
        step_diff = np.max([0, steps - self.l_start])

        return self.end + np.max([
            0,
            start_diff * (self.endt - step_diff) / self.endt])
