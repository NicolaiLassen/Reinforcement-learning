import os

import torch
from matplotlib import pyplot as plt

from project.agents.PPO import PPO
from project.models.actorCritic import ActorCritic
from project.environment.environment import EnvWrapper
from project.models.Actor import Actor

# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

if __name__ == '__main__':
    # Hyperparameters
    step_size = 4

    # Environment initialization
    env = EnvWrapper('procgen:procgen-starpilot-v0', step_size)

    actor = Actor(state_dim=64, action_dim=env.env.action_space.n)
    ## TODO
    critic = Actor(state_dim=64, action_dim=1)

    episodes = 1000
    learning_rate = 0.005

    optimizer = torch.optim.Adam([
        {'params': actor.parameters(), 'lr': learning_rate},
        {'params': critic.parameters(), 'lr': learning_rate}
    ])

    obs = env.reset()
    for i in range(episodes):
        agent = ActorCritic(env, actor, critic, optimizer)
        obs, reward, done, _ = env.step(env.env.action_space.sample())
        #agent.act(obs)
        if done:
            break
