import copy
from collections import deque

import torch
from torch.utils.tensorboard import SummaryWriter
from unityagents import UnityEnvironment
import numpy as np

from agent_config import AgentConfig
from dqn_agent import Agent
from models import NetworkType
from replay_buffers import BufferType


def test_agent(env, agent, brain_name, skip_frames):
    for i in range(5):
        env_info = env.reset(train_mode=False)[brain_name]  # reset the environment
        state = env_info.vector_observations[0]
        score = 0
        for j in range(1000):
            action = agent.act(state)
            for frames in range(skip_frames + 1):
                env_info = env.step(action)[brain_name]

            state = env_info.vector_observations[0]  # get the next state
            reward = env_info.rewards[0]  # get the reward
            done = env_info.local_done[0]  # see if episode has finished
            score += reward
            if done:
                break
        print("Score: {}".format(score))


def train_agent(env, agent, brain_name, n_episodes=1600, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995,
                summary_writer=None):
    """Deep Q-Learning.

    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores

    # save weight when the score is higher that current max
    max_score = 13.
    max_score_episode = 0

    eps = eps_start  # initialize epsilon
    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
        state = env_info.vector_observations[0]  # get the current state
        score = 0
        losses = []
        for t in range(max_t):
            action = agent.act(state, eps)
            for frames in range(agent.skip_frames + 1):
                env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]  # get the next state
            reward = env_info.rewards[0]  # get the reward
            done = env_info.local_done[0]  # see if episode has finished
            # next_state, reward, done, _ = env.step(action)
            loss = agent.step(state, action, reward, next_state, done)
            if loss:
                losses.append(loss)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score
        eps = max(eps_end, eps_decay * eps)  # decrease epsilon

        mean_score = np.mean(scores_window)
        if mean_score > max_score + 0.3:
            max_score = mean_score
            torch.save(agent.qnetwork_local.state_dict(), 'weights_final/{}.pth'.format(agent.config))
            max_score_episode = i_episode - 100
            print('\nNew max score. Weights saved {:d} episodes!\tAverage Score: {:.2f}'.format(max_score_episode,
                                                                                                mean_score))

        if summary_writer is not None:
            writer.add_scalar('Avg_Loss', np.mean(losses), i_episode)
            writer.add_scalar('Avg_Reward', np.mean(scores_window), i_episode)

        print('\rEpisode {}\tAverage Score: {:.2f} Losses: {:.2f}'.format(i_episode, np.mean(scores_window),
                                                                          np.mean(losses)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f} Losses: {:.2f}'.format(i_episode, np.mean(scores_window),
                                                                              np.mean(losses)))

    return scores


def generate_agent_configs():
    # cfg_1 = AgentConfig(network_type=NetworkType.DQN, buffer_type=BufferType.NORMAL, lr=3e-4, gamma=0.95,
    #                     double_dqn=True)
    cfg_2 = AgentConfig(network_type=NetworkType.DQN, buffer_type=BufferType.NORMAL, lr=3e-4, gamma=0.95,
                        double_dqn=False)
    cfg_3 = AgentConfig(network_type=NetworkType.DUEL_DQN, buffer_type=BufferType.NORMAL, lr=3e-4, gamma=0.95,
                        double_dqn=False)
    cfg_4 = AgentConfig(network_type=NetworkType.DUEL_DQN, buffer_type=BufferType.NORMAL, lr=3e-4, gamma=0.95,
                        double_dqn=True)

    return [ cfg_2, cfg_3, cfg_4]


if __name__ == '__main__':
    env = UnityEnvironment(file_name="Banana_Linux/Banana.x86")
    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # number of agents in the environment
    print('Number of agents:', len(env_info.agents))

    # number of actions
    action_size = brain.vector_action_space_size
    print('Number of actions:', action_size)

    # examine the state space
    state = env_info.vector_observations[0]
    print('States look like:', state)
    state_size = len(state)
    print('States have length:', state_size)

    configs = generate_agent_configs()
    for agent_cfg in configs:
        agent = Agent(state_size=state_size, action_size=action_size, agent_config=agent_cfg)
        writer = SummaryWriter("runs_final/{}".format(agent_cfg))
        # agent.qnetwork_local.load_state_dict(torch.load('{}.pth'.format(agent_cfg)))
        # test_agent(env, agent, brain_name, agent_cfg.skip_frames)
        train_agent(env, agent, brain_name, summary_writer=writer)
        # torch.save(agent.qnetwork_local.state_dict(), 'best_weight/{}_2.pth'.format(agent_cfg))
    # test_agent(env, agent, brain_name)

    env.close()
