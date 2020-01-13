from collections import namedtuple

from dqn_agent import Agent
from models import NetworkType
from replay_buffers import BufferType

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64  # minibatch size
GAMMA = 0.99  # discount factor
TAU = 1e-3  # for soft update of target parameters
LR = 5e-4  # learning rate
UPDATE_EVERY = 4  # how often to update the network
SKIP_FRAMES = 0  #
DOUBLE_DQN = False

ALPHA = 0.6
BETA_START = 0.4
BETA_END = 1.0
BETA_MAX_STEPS = 10000
BUFFER_EPSILON = 0.01
SEED = 0


class AgentConfig:

    def __init__(self, network_type=NetworkType.DQN, buffer_type=BufferType.NORMAL, lr=LR, gamma=GAMMA, tau=TAU,
                 update_every=UPDATE_EVERY,
                 seed=SEED, skip_frames=SKIP_FRAMES, buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE,
                 buffer_epsilon=BUFFER_EPSILON, alpha=ALPHA, beta_start=BETA_START, beta_end=BETA_END,
                 beta_max_steps=BETA_MAX_STEPS, double_dqn=DOUBLE_DQN):
        self.network_type = network_type
        self.buffer_type = buffer_type
        self.lr = lr
        self.double_dqn = double_dqn
        self.gamma = gamma
        self.tau = tau
        self.update_every = update_every
        self.seed = seed
        self.skip_frames = skip_frames
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.buffer_epsilon = buffer_epsilon
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_max_steps = beta_max_steps

    def __str__(self):
        double = "" if not self.double_dqn else "DOUBLE_"
        return "{}{}_buffer_{}_lr_{}_skip_{}_alpha_{:.2f}_beta_start_{:.2f}_end_{:.2f}_steps_{}_tau_{}_batch_size_{}_gamma_{:.2f}".format(
            double, self.network_type.name, self.buffer_type.name, self.lr,
            self.skip_frames, self.alpha, self.beta_start, self.beta_end, self.beta_max_steps, self.tau,
            self.batch_size, self.gamma)
