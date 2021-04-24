import gym
import torch
from torch.distributions import Categorical

from project.agents.BaseAgent import BaseAgent
from project.models.policy_models import PolicyModelEncoder, PolicyModel


class PPOAgent(BaseAgent):
    def __init__(self, env: gym.Env, actor, critic, optimizer=None):
        self.env = env
        self.actor = actor
        self.critic = critic
        self.optimizer = optimizer

    def act(self, state):
        pass

    def eval(self):
        pass

    def train(self, num_episodes=5, num_steps=100):
        for i in range(num_episodes):

            s1 = self.env.reset()
            SAR = []

            for j in range(num_steps):
                s = s1
                s = torch.from_numpy(s).permute(2, 1, 0).to(torch.float)
                act_probs = self.actor(s)
                act_dist = Categorical(act_probs)
                act = act_dist.sample()
                s1, r, done, _ = self.env.step(act)
                if done:
                    SAR.append((s, act, r, 0))
                    s1 = self.env.reset()
                    continue
                r_net = self.critic(s)
                SAR.append((s, act, r, r_net))
            print(SAR)


if __name__ == "__main__":
    # env = EnvWrapper('procgen:procgen-starpilot-v0', 1)
    env = gym.make('procgen:procgen-starpilot-v0')

    actor = PolicyModelEncoder(3, 64, 64, env.action_space.n)
    critic = PolicyModel(3, 64, 64, 1)

    agent = PPOAgent(env, actor, critic)
    agent.train()
