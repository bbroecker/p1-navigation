from enum import Enum

import numpy as np
import random
from collections import namedtuple, deque

from models import DQNNetwork, DQNDuelingNetwork, NetworkType

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from replay_buffers import BufferType, StandardBuffer, PriorityBuffer
from sum_tree import SumTree


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, agent_config):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(agent_config.seed)
        self.config = agent_config
        seed = agent_config.seed
        self.skip_frames = agent_config.skip_frames
        # Q-Network
        if agent_config.network_type == NetworkType.DQN:
            self.qnetwork_local = DQNNetwork(state_size, action_size, seed).to(device)
            self.qnetwork_target = DQNNetwork(state_size, action_size, seed).to(device)
        elif agent_config.network_type == NetworkType.DUEL_DQN:
            self.qnetwork_local = DQNDuelingNetwork(state_size, action_size, seed).to(device)
            self.qnetwork_target = DQNDuelingNetwork(state_size, action_size, seed).to(device)

        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=agent_config.lr)
        self.criterion = nn.MSELoss()
        # Replay memory
        if agent_config.buffer_type == BufferType.NORMAL:
            self.buffer = StandardBuffer(action_size, agent_config)
        elif agent_config.buffer_type == BufferType.PRIORITY:
            self.buffer = PriorityBuffer(action_size, agent_config)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        self.gamma = agent_config.gamma
        self.tau = agent_config.tau
        self.update_every = agent_config.update_every
        self.batch_size = agent_config.batch_size

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        state_dev = torch.from_numpy(state).float().unsqueeze(0).to(device)
        next_state_dev = torch.from_numpy(next_state).float().unsqueeze(0).to(device)
        q_new = reward + self.gamma * self.qnetwork_target.forward(next_state_dev).detach().max(1)[0] * (1 - done)
        q_old = self.qnetwork_local.forward(state_dev).data.cpu().numpy()[0][action]
        error = abs(q_new.cpu().detach().numpy() - q_old)
        self.buffer.add((state, action, reward, next_state, done), error)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.buffer) > self.batch_size:
                return self.learn()
        return None

    def buffer_size(self):
        return len(self.buffer)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        experiences, idxs, is_weights = self.buffer.sample()
        states, actions, rewards, next_states, dones = experiences

        if self.config.double_dqn:
            argmax_actions = self.qnetwork_local(next_states).detach().max(1)[1].unsqueeze(1)
            q_max = self.qnetwork_target(next_states).gather(1, argmax_actions)
        else:
            q_max = self.qnetwork_target.forward(next_states).detach().max(1)[0].unsqueeze(1)

        q_targets = rewards + (self.gamma * q_max * (1 - dones))
        q_expected = self.qnetwork_local.forward(states).gather(1, actions)

        errors = torch.abs(q_targets - q_expected).data.cpu().numpy()
        for idx, error in zip(idxs, errors):
            self.buffer.update(idx, error)

        self.optimizer.zero_grad()
        # if prioritised buffer is active the best_weight will effect the update
        loss = (torch.FloatTensor(is_weights).to(device) * F.mse_loss(q_expected, q_targets).squeeze())
        loss = loss.mean()

        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)

        return loss.detach().cpu().numpy()

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): best_weight will be copied from
            target_model (PyTorch model): best_weight will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


