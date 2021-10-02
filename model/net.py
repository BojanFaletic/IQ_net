import torch.optim as optim
from .layers import LinearMask
import torch
import torch.nn as nn
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

    def predict(self):
        prediction_list = []
        expected_list = []

        with torch.no_grad():
            for m in range(self.data_width - self.mask_size + 1):
                mm = np.arange(self.mask_size) + m
                out = self.net(self.data, mm)
                expected = self.data[mm]

                prediction_list.append(out.detach().item())
                expected_list.append(expected.detach().item())
        prediction_list = [round(x, 2) for x in prediction_list]
        return prediction_list, expected_list

    def eval(self):
        accuracy = 0
        with torch.no_grad():
            for m in range(self.data_width - self.mask_size + 1):
                mm = np.arange(self.mask_size) + m
                out = self.net(self.data, mm)
                expected = self.data[mm]

                accuracy += Net.is_close(out, expected)
        return accuracy / self.data_width

    def train(self):
        criterion = nn.MSELoss()
        optimizer = optim.SGD(self.net.parameters(), lr=1e-2,
                              momentum=0.4, nesterov=True)
        epochs = 100
        print_interval = 10
        single_mask_train = 10

        loss_hist = []

        for e in range(epochs):
            acc_loss = 0
            for m in range(self.data_width-self.mask_size + 1):
                mm = np.arange(self.mask_size) + m
                # learn few times
                for i in range(single_mask_train):
                    optimizer.zero_grad()

                    out = self.net(self.data, mm)
                    expected = self.data[mm]

                    loss = criterion(out, expected)
                    loss.backward()

                    acc_loss += loss.detach().item()

                    optimizer.step()

            loss_hist.append(acc_loss)
            if e % print_interval == 0:
                acc = self.eval() * 100
                print(f'ep: {e : 2d}, loss: {acc_loss :.2f}, acc: {acc :.2f}%')
        return loss_hist
