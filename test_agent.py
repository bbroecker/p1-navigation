from unityagents import UnityEnvironment

from agent_config import AgentConfig
from dqn_agent import Agent
from models import NetworkType
from replay_buffers import BufferType
from utils.train_utils import train_agent, load_weights, test_agent

if __name__ == '__main__':
    env = UnityEnvironment(file_name="Banana_Linux/Banana.x86")
    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # number of actions
    action_size = brain.vector_action_space_size
    print('Number of actions:', action_size)

    # examine the state space
    state = env_info.vector_observations[0]
    state_size = len(state)
    print('States have length:', state_size)

    agent_cfg = AgentConfig(network_type=NetworkType.DQN, buffer_type=BufferType.NORMAL, lr=3e-4, gamma=0.95,
                            double_dqn=True)
    agent = Agent(state_size=state_size, action_size=action_size, agent_config=agent_cfg)
    load_weights(agent, 'best_weight/DDQN.pth')
    # agent.qnetwork_local.load_state_dict(torch.load('{}.pth'.format(agent_cfg)))
    test_agent(env, agent, brain_name, agent_cfg.skip_frames)
    # train_agent(env, agent, "weights/new_weight.pth", brain_name)
    # torch.save(agent.qnetwork_local.state_dict(), 'best_weight/{}_2.pth'.format(agent_cfg))
    # test_agent(env, agent, brain_name)
