import torch
import torch.nn as nn


class LinearMaskActivation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, mask):
        m_t = torch.tensor(mask)

        ctx.save_for_backward(input, weight, bias, m_t)

        w_cp = weight.clone()
        w_cp[:, mask] = 0

        out = input @ w_cp.t() + bias
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias, mask = ctx.saved_tensors

        w_cp = weight.clone()
        w_cp[:, mask] = 0

        # reshape grad output to 2d matrix
        grad_output = grad_output.unsqueeze(0)
        input = input.unsqueeze(0)

        grad_input = grad_output @ w_cp

        grad_weight = grad_output.t() @ input
        grad_weight[:, mask] = 0

        grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias, None


class LinearMask(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(LinearMask, self).__init__()
        self.fc = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, input, mask):
        return LinearMaskActivation.apply(input, self.fc.weight,
                                          self.fc.bias, mask)
