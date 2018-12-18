import numpy as np


class AVGMeter(object):

    def __init__(self, precision: int = 6, log_freq=None):
        self.precision = precision
        self.log_freq = log_freq
        self.sum = 0.0
        self.count = 0
        self.values = []

    def append(self, x):
        self.values.append(x)
        self.sum += x
        self.count += 1

    def reset(self):
        self.sum = 0.0
        self.count = 0
        self.values = []

    def lasts_avg(self, last_size=10):
        if last_size <= len(self.values):
            last_mean = np.mean(self.values[-last_size:])
        else:
            last_mean = np.mean(self.values[0:])
        return float(round(last_mean, self.precision))

    @property
    def avg(self) -> float:
        return round(self.sum / self.count, self.precision)

    @property
    def avgl(self):
        return self.lasts_avg()

    @property
    def last(self):
        if len(self.values) > 0:
            return self.values[-1]

    @property
    def vlen(self):
        return len(self.values)
