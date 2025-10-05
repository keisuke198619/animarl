### Code for Gromov-Wasserstein Imitation Learning, Arnaud Fickinger, 2022
# Copyright (c) Meta Platforms, Inc. and affiliates.

import numpy as np
import torch
from torch import nn
import os
import random


class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


class train_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(True)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data +
                                (1 - tau) * target_param.data)


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def make_dir(*path_parts):
    dir_path = os.path.join(*path_parts)
    try:
        os.mkdir(dir_path)
    except OSError:
        pass
    return dir_path


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


class MLP(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 hidden_depth,
                 output_mod=None):
        super().__init__()
        self.trunk = mlp(input_dim, hidden_dim, output_dim, hidden_depth,
                         output_mod)
        self.apply(weight_init)

    def forward(self, x):
        return self.trunk(x)


def mlp(input_dim, hidden_dim, output_dim, hidden_depth, tanh=False, output_mod=None):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim),nn.Tanh()]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
        if tanh:
            mods.append(nn.Tanh())
    if output_mod is not None:
        mods.append(output_mod)
    trunk = nn.Sequential(*mods)
    return trunk

class RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, hidden_depth, tanh=False, output_mod=None):
        super(RNN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.hidden_depth = hidden_depth
        
        # Initial transformation
        self.initial_transform = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # RNN layers
        self.rnn_layers = nn.ModuleList([
            nn.RNN(hidden_dim, hidden_dim, batch_first=True) for _ in range(hidden_depth)
        ])
        
        # Output transformation
        self.output_transform = nn.Linear(hidden_dim, output_dim)
        if tanh:
            self.output_transform = nn.Sequential(self.output_transform, nn.Tanh())
        if output_mod is not None:
            self.output_transform.add_module('output_mod', output_mod)

    def forward(self, x, hidden_states):
        # Initial transformation
        x = self.initial_transform(x)

        # Apply RNN layers
        new_hidden_states = []
        for i, rnn_layer in enumerate(self.rnn_layers):
            x, h = rnn_layer(x.unsqueeze(1), hidden_states[i])
            x = x.squeeze(1)
            new_hidden_states.append(h)

        # Apply output transformation
        x = self.output_transform(x.squeeze(1))

        return x, torch.stack(new_hidden_states).squeeze(1)

'''class RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, hidden_depth,output_mod=None):
        super().__init__()
        self.trunk = rnn(input_dim, hidden_dim, output_dim, hidden_depth,
                         output_mod)
        self.apply(weight_init)
        
    def forward(self, x):
        return self.trunk(x)

def rnn(input_dim, hidden_dim, output_dim, hidden_depth, tanh=False, output_mod=None):
    mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
    for i in range(hidden_depth):
        mods += [nn.RNN(hidden_dim, hidden_dim, hidden_depth, batch_first=True)]
    mods.append(nn.Linear(hidden_dim, output_dim))
    if tanh:
        mods.append(nn.Tanh())
    if output_mod is not None:
        mods.append(output_mod)
    trunk = nn.Sequential(*mods)
    return trunk'''
   


def to_np(t):
    if t is None:
        return None
    elif t.nelement() == 0:
        return np.array([])
    else:
        return t.cpu().detach().numpy()
