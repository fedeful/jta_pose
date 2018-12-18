import numpy as np
import sys
import time
from datetime import datetime


class ProgressBar(object):
    def __init__(self, bar_len):
        self.bar_len = bar_len

    def clear(self):
        pass

    def update(self, progress, info_steps, info_losses):

        if progress == 0:
            block = 0
        else:
            block = int(round(self.bar_len * progress))

        # text = u"\r[{0}] Epoch: {1:s} ◖{2}◗ {3:.2f}% | ".format(datetime.now().strftime("%Y-%m-%d@%H:%M"),
        #                                                        info_steps,
        #                                                        '█' * block + '┈' * (self.bar_len - block),
        #                                                        progress * 100)

        text = "\r[{0}] Epoch: {1:s} | {2:.2f}% | ".format(datetime.now().strftime("%Y-%m-%d@%H:%M"), info_steps,
                                                                progress * 100)

        text += info_losses
        sys.stdout.write(text)
        sys.stdout.flush()
