import os

import torch
from matplotlib import pyplot as plt

from project.environment.EnvWrapper import EnvWrapper
from project.models.policy_models import PolicyModel

# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

if __name__ == '__main__':
    # Hyperparameters
    step_size = 4

    # Environment initialization
    env = EnvWrapper('procgen:procgen-starpilot-v0', step_size)

    actor = PolicyModel(width=64, action_dim=env.env.action_space.n)
    ## TODO could be smaller network
    critic = PolicyModel(width=64, action_dim=1)

    episodes = 1000
    learning_rate = 0.005

    optimizer = torch.optim.Adam([
        {'params': actor.parameters(), 'lr': learning_rate},
        {'params': critic.parameters(), 'lr': learning_rate}
    ])

    obs = env.reset()
    agent = ActorCritic(env, actor, critic, optimizer)

    for i in range(episodes):

        obs, reward, done, _ = env.step(env.env.action_space.sample())
        #agent.act(obs)
        if done:
            break
