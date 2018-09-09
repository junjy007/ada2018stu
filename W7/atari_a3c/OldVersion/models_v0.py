import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import weights_init


class StateRepresentor(torch.nn.Module):
    def __init__(self, in_channels):
        super(StateRepresentor, self).__init__()
        conv1 = nn.Conv2d(in_channels, 32, 3, stride=2, padding=1)
        conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv_layers = nn.Sequential(
            conv1, nn.ELU(),
            conv2, nn.ELU(),
            conv3, nn.ELU(),
            conv4, nn.ELU())
        self.apply(weights_init)

    def forward(self, state_tenor):
        num_samples = state_tenor.shape[0]
        feats = self.conv_layers(state_tenor)
        feats = feats.view(num_samples, -1)
        return feats


class Actor(torch.nn.Module):
    def __init__(self, feat_num, action_num):
        super(Actor, self).__init__()
        self.h_linear = nn.Linear(feat_num, 256)
        self.a_linear = nn.Linear(256, action_num)

    def forward(self, feats):
        h = F.elu(self.h_linear(feats))
        lp = self.a_linear(h)
        return lp


class Agent(torch.nn.Module):
    def __init__(self, input_channels, feat_num, actions):
        super(Agent, self).__init__()
        self.repnet = StateRepresentor(input_channels)
        self.actor = Actor(feat_num, actions)

    def forward(self, state_tensor):
        return self.actor(self.repnet(state_tensor))






         # self.critic_linear = nn.Linear(256, 1)
        # self.actor_linear = nn.Linear(256, num_outputs)

        # self.apply(weights_init)
        # self.actor_linear.weight.data = normalized_columns_initializer(
        #     self.actor_linear.weight.data, 0.01)
        # self.actor_linear.bias.data.fill_(0)
        # self.critic_linear.weight.data = normalized_columns_initializer(
        #     self.critic_linear.weight.data, 1.0)
        # self.critic_linear.bias.data.fill_(0)

        # self.lstm.bias_ih.data.fill_(0)
        # self.lstm.bias_hh.data.fill_(0)

        # self.train()
        # if USE_CUDA:
        return lp