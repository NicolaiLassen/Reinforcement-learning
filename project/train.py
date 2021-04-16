import os

import torch
from matplotlib import pyplot as plt

from project.agents.PPO import PPO
from project.environment.environment import EnvWrapper
from project.models.ActorCritic import ActorCritic

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

if __name__ == '__main__':
    # Hyperparameters
    step_size = 4

    # Environment initialization
    env = EnvWrapper('procgen:procgen-starpilot-v0', step_size)

    actor = ActorCritic(state_dim=64, action_dim=env.env.action_space.n, action_mask="?")
    critic = ActorCritic(state_dim=64, action_dim=env.env.action_space.n, action_mask="?")

    episodes = 1000
    learning_rate = 0.005

    optimizer = torch.optim.Adam([
        {'params': actor.parameters(), 'lr': learning_rate},
        {'params': critic.parameters(), 'lr': learning_rate}
    ])

    obs = env.reset()
    for i in range(episodes):
        agent = PPO(actor, critic, optimizer)
        obs, reward, done, _ = env.step(env.env.action_space.sample())
        agent.act(obs)