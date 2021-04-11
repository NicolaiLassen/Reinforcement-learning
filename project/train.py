import gym
import matplotlib.pyplot as plt
import torch
from gym.wrappers import FrameStack
from environment.environment import EnvWrapper
from project.agents.PPO import PPO
from project.models.ActorCritic import ActorCritic

if __name__ == '__main__':
    # Hyperparameters
    step_size = 4

    # Environment initialization
    env = EnvWrapper('procgen:procgen-starpilot-v0', step_size)

    obs, reward = env.step(env.env.action_space.sample())
    print(obs.shape, reward)
    while(True):

#if __name__ == '__main__':
#
#    learning_rate = 0.005
#    episodes = 1000
#
#    model = ActorCritic()
#    optimizer = torch.optim.Adam([
#        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
#        {'params': self.policy.critic.parameters(), 'lr': lr_critic}
#    ])
#
#    agent = PPO(model, 1, optimizer)
#
#    # create traning loop
#    # env.render(mode="rgb_array")
#
#    obs = env.reset()
#    for i in range(episodes):
#        # use screen to generalize problems
#        screen = env.render(mode="rgb_array")
#        agent.act(screen)

