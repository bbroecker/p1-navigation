from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F

class NetworkType(Enum):
    DQN = 0
    DUEL_DQN = 1

class DQNNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, hidden_1_num=64, hidden_2_num=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(DQNNetwork, self).__init__()
        if seed is not None:
            self.seed = torch.manual_seed(seed)
        self.input_layer = nn.Linear(state_size, hidden_1_num)
        self.hidden_layer = nn.Linear(hidden_1_num, hidden_2_num)
        self.output_layer = nn.Linear(hidden_2_num, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.input_layer(state))
        x = F.relu(self.hidden_layer(x))
        return self.output_layer(x)


class DQNDuelingNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, val_hidden_1_num=64, val_hidden_2_num=64, adv_hidden_1_num=64,
                 adv_hidden_2_num=64, max_action=False):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(DQNDuelingNetwork, self).__init__()
        if seed is not None:
            self.seed = torch.manual_seed(seed)
        self.input_layer_val = nn.Linear(state_size, val_hidden_1_num)
        self.hidden_layer_val = nn.Linear(val_hidden_1_num, val_hidden_2_num)
        self.output_layer_val = nn.Linear(val_hidden_2_num, action_size)

        self.input_layer_adv = nn.Linear(state_size, adv_hidden_1_num)
        self.hidden_layer_adv = nn.Linear(adv_hidden_1_num, adv_hidden_2_num)
        self.output_layer_adv = nn.Linear(adv_hidden_2_num, 1)
        self.max_action = max_action
        self.action_size = action_size


    def forward(self, state):
        """Build a network that maps state -> action values."""
        batch_size = state.size(0)
        val = F.relu(self.input_layer_val(state))
        val = F.relu(self.hidden_layer_val(val))
        val = self.output_layer_val(val)
        val = val.expand(batch_size, self.action_size)

        adv = F.relu(self.input_layer_adv(state))
        adv = F.relu(self.hidden_layer_adv(adv))
        adv = F.relu(self.output_layer_adv(adv))

        if self.max_action:
            result = val + (adv - adv.max(1)[0].unsqueeze(1).expand(batch_size, self.action_size))
        else:
            result = val + adv - adv.mean()

        return result
