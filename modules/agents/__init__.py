from .rnn_agent import RNNAgent
from .mlp_agent import MLPAgent

REGISTRY = {}


REGISTRY["rnn"] = RNNAgent
REGISTRY["mlp"] = MLPAgent


# -----------------------------------
### Code for Gromov-Wasserstein Imitation Learning, Arnaud Fickinger, 2022
# Copyright (c) Meta Platforms, Inc. and affiliates.
import abc


class Agent(object):
    def reset(self):
        """For state-full agents this function performs reseting at the beginning of each episode."""
        pass

    @abc.abstractmethod
    def train(self, training=True):
        """Sets the agent in either training or evaluation mode."""

    @abc.abstractmethod
    def update(self, replay_buffer, logger, step):
        """Main function of the agent that performs learning."""

    @abc.abstractmethod
    def act(self, obs, sample=False):
        """Issues an action given an observation."""
