import torch.optim as optim
from .layers import LinearMask
import torch
import numpy as np


class Net:
    def __init__(self, data_width, mask_size):
        self.data_width = data_width
        self.mask_size = mask_size

        expand_sz = data_width // 2
        self.data = torch.tensor([0, 1] * expand_sz, dtype=torch.float)
        self.net = LinearMask(data_width, mask_size)

    @staticmethod
    def is_close(a, b, atol=0.1):
        return np.isclose(a, b, atol=atol).sum()

    def eval(self):
        accuracy = 0
        with torch.no_grad():
            for m in range(self.data_width - self.mask_size):
                mm = np.arange(self.mask_size) + m
                out = self.net(self.data, mm)
                expected = self.data[mm]

                accuracy += Net.is_close(out, expected)
        return accuracy / self.data_width



