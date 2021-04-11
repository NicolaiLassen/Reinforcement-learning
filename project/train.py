import torch

from project.agents.PPO import PPO
from project.models.ActorCritic import ActorCritic
import gym
env = gym.make('procgen:procgen-coinrun-v0')

if __name__ == '__main__':

    learning_rate = 0.005
    episodes = 1000

    model = ActorCritic()
    optimizer = torch.optim.Adam([
        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
        {'params': self.policy.critic.parameters(), 'lr': lr_critic}
    ])

    agent = PPO(model, 1, optimizer)

    # create traning loop
    # env.render(mode="rgb_array")

    obs = env.reset()
    for i in range(episodes):
        # use screen to generalize problems
        screen = env.render(mode="rgb_array")
        agent.act(screen)

