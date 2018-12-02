import torch.nn as nn
import torch
import numpy as np

from collections import OrderedDict


def create_grad_array(layer):
    layer_weight_grad = torch.zeros(
        layer.weight.grad.size()
    ).float()
    layer_bias_grad = torch.zeros(
        layer.bias.grad.size()
    ).float()

    return (layer_weight_grad, layer_bias_grad)


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


class TorchConvnet(nn.Module):

    def __init__(self, input_dims, n_actions):
        super(TorchConvnet, self).__init__()
        self.input_dims = input_dims
        self.n_actions = n_actions

        self.conv1 = nn.Conv2d(4, 32, 8, stride=4, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)

        x = torch.zeros((1,) + self.input_dims)
        x = self.conv3.forward(
                self.conv2.forward(
                    self.conv1.forward(x)
                )
            )

        self.r2 = Reshape(np.prod(x.shape))

        self.fc1 = nn.Linear(np.prod(x.shape), 512, True)
        self.fc2 = nn.Linear(512, n_actions, True)

        x = self.forward(np.zeros((1,) + self.input_dims))
        self.backward(x, x)

        self._build_grad_arrays()
        self._build_dw()

    def _build_dw(self):
        self.dw = [
            (self.conv1.weight.grad, self.conv1.bias.grad),
            (self.conv2.weight.grad, self.conv2.bias.grad),
            (self.conv3.weight.grad, self.conv3.bias.grad),
            (self.fc1.weight.grad, self.fc1.bias.grad),
            (self.fc2.weight.grad, self.fc2.bias.grad),
        ]

    def _build_grad_arrays(self):

        def array():
            return [
                create_grad_array(self.conv1),
                create_grad_array(self.conv2),
                create_grad_array(self.conv3),
                create_grad_array(self.fc1),
                create_grad_array(self.fc2)
            ]

        self.g = array()
        self.g2 = array()
        self.tmp = array()
        self.deltas = array()

    def forward(self, x):

        if len(x.shape) == 4:
            return self._forward_batch(x)

        return self._forward_single(
            torch.from_numpy(x).float())

    def _forward_single(self, x):
        assert x.shape == (1,) + self.input_dims
        x = nn.ReLU().forward(self.conv1.forward(x))
        x = nn.ReLU().forward(self.conv2.forward(x))
        x = nn.ReLU().forward(self.conv3.forward(x))
        x = self.r2.forward(x)
        x = nn.ReLU().forward(self.fc1.forward(x))

        return self.fc2.forward(x)

    def _forward_batch(self, x):
        x_all = torch.zeros(x.shape[0], self.n_actions)
        for i in range(x.shape[0]):
            x_i = torch.from_numpy(
                np.expand_dims(x[i], axis=0)
            ).float()
            x_all[i] = self._forward_single(x_i)
        return x_all

    def backward(self, out, targets):
        self.zero_grad()
        out.backward(targets)
        self._build_dw()

    def apply_update(self, lr, wc):

        for (g, dw) in zip(self.g, self.dw):
            g[0].mul(0.95).add(0.05, dw[0])
            g[1].mul(0.95).add(0.05, dw[1])

        # this is disgusting but pytorch is abit weird.
        # torch.mul(Tensor a, Tensor b, Tensor out)
        # errors becuase its an invalid combination of
        # elements.
        # EVEN THOUGH THE DOCS AND ERROR MSG SAYS ITS FINE
        i = 0
        for i in range(len(self.tmp)):
            self.tmp[i] = (self.dw[i][0].mul(self.dw[i][0]),
                           self.dw[i][1].mul(self.dw[i][1]))

        for (g2, tmp) in zip(self.g2, self.tmp):
            g2[0].mul(0.95).add(0.05, tmp[0])
            g2[1].mul(0.95).add(0.05, tmp[1])

        i = 0
        for i in range(len(self.tmp)):
            self.tmp[i] = (self.g[i][0].mul(self.g[i][0]),
                           self.g[i][1].mul(self.g[i][1]))

        for (tmp, g2) in zip(self.tmp, self.g2):
            tmp[0].mul(-1)
            tmp[1].mul(-1)
            tmp[0].add(g2[0])
            tmp[1].add(g2[1])
            tmp[0].add(0.01)
            tmp[1].add(0.01)
            tmp[0].sqrt()
            tmp[1].sqrt()

        for i in range(len(self.g)):
            self.deltas[i][0].mul(0)
            torch.addcdiv(
                self.deltas[i][0],
                lr, self.dw[i][0], self.tmp[i][0]
            )
            self.deltas[i][1].mul(0)
            torch.addcdiv(
                self.deltas[i][1],
                lr, self.dw[i][1], self.tmp[i][1]
            )

        i = 0
        for w, b in zip(*[self.parameters()]*2):
            w.add(self.deltas[i][0])
            b.add(self.deltas[i][1])
            i += 1


def build_network_torch_class(input_dims, n_actions):

    return TorchConvnet(input_dims, n_actions)


def build_network_torch(input_dims, n_actions):
    model = nn.Sequential(OrderedDict([
        #('r1', Reshape(*input_dims)),
        ('conv1', nn.Conv2d(4, 32, 8, stride=4, padding=2)),
        ('relu1', nn.ReLU()),
        ('conv2', nn.Conv2d(32, 64, 4, stride=2)),
        ('relu2', nn.ReLU()),
        ('conv3', nn.Conv2d(64, 64, 3, stride=1)),
        ('relu3', nn.ReLU())
    ]))

    out_shape = model.forward(torch.zeros((1,) + input_dims)).shape
    nel = np.prod(out_shape)
    model.add_module('r2', Reshape(nel))
    model.add_module('fc1', nn.Linear(nel, 512))
    model.add_module('relu4', nn.ReLU())
    model.add_module('fc2', nn.Linear(512, n_actions))

    return model
