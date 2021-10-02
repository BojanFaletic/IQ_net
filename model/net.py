import torch.optim as optim
from .layers import LinearMask
import torch
import torch.nn as nn
import numpy as np


class Net:
    def __init__(self, data_width, mask_size, pattern=[-1, -1, 1]):
        self.data_width = data_width
        self.mask_size = mask_size

        msg = "Data width must be divisible with pattern"
        assert data_width % len(pattern) == 0, msg

        expand_sz = data_width // len(pattern)
        self.data = torch.tensor(pattern * expand_sz, dtype=torch.float)

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

                prediction_list.append(out.detach().numpy())
                expected_list.append(expected.detach().numpy())
        prediction_list = [np.round(x, 2) for x in prediction_list]
        return prediction_list, expected_list

    def eval(self):
        prediction, expectation = self.predict()
        accuracy = np.sum([self.is_close(a, b) for a, b in
                           zip(prediction, expectation)])

        total_items = len(prediction) * self.mask_size
        return accuracy / total_items

    def train(self):
        criterion = nn.MSELoss()
        optimizer = optim.SGD(self.net.parameters(), lr=1e-3,
                              momentum=0.4, nesterov=True)
        epochs = 200
        print_interval = 10
        single_mask_train = 100

        loss_hist = []

        for e in range(epochs):
            acc_loss = 0
            for m in range(self.data_width - self.mask_size + 1):
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
                if acc > 99.99:
                    break
        return loss_hist
