import torch.nn as nn
import torch
import numpy as np
import logging

mylogger = logging.getLogger("testSA2")


class smallNet(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dims=None):
        super().__init__()
        if hidden_dims == None:
            hidden_dims = [5, 5]
        nn_dims = [in_dim] + hidden_dims + [out_dim]
        modules = []
        for i in range(len(nn_dims) - 1):
            if i == len(nn_dims) - 2:
                modules.append(
                    nn.Sequential(nn.Linear(nn_dims[i], nn_dims[i + 1])))
            else:
                modules.append(
                    nn.Sequential(nn.Linear(nn_dims[i], nn_dims[i + 1]),
                                  nn.ReLU()))
                #nn.Tanh()))
        self.net = nn.Sequential(*modules)
        # self.net.apply(self.weights_init_uniform_rule)

    def forward(self, x):
        return self.net(x)

    def call(self, x):
        with torch.no_grad():
            return self.forward(torch.tensor(x, dtype=torch.float32)).numpy()

    @staticmethod
    def weights_init_uniform_rule(m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            n = m.in_features
            y = 1.0 / np.sqrt(n)
            m.weight.data.uniform_(-y, y)
            m.bias.data.fill_(0)

    def reset_parameters(self):
        self.net.apply(self.weights_init_uniform_rule)


class anotherSmallNet(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dims=None):
        super().__init__()
        if hidden_dims == None:
            hidden_dims = [32]
        nn_dims = [in_dim] + hidden_dims + [out_dim]
        modules = []
        for i in range(len(nn_dims) - 1):
            if i == len(nn_dims) - 2:
                modules.append(
                    nn.Sequential(nn.Linear(nn_dims[i], nn_dims[i + 1])))
            else:
                modules.append(
                    nn.Sequential(nn.Linear(nn_dims[i], nn_dims[i + 1]),
                                  nn.Tanh()))
                #nn.Tanh()))
        self.net = nn.Sequential(*modules)
        # self.net.apply(self.weights_init_uniform_rule)

    def forward(self, x):
        return self.net(x)

    def call(self, x):
        with torch.no_grad():
            return self.forward(torch.tensor(x, dtype=torch.float32)).numpy()

    @staticmethod
    def weights_init_uniform_rule(m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            n = m.in_features
            y = 1.0 / np.sqrt(n)
            m.weight.data.uniform_(-y, y)
            m.bias.data.fill_(0)

    def reset_parameters(self):
        self.net.apply(self.weights_init_uniform_rule)