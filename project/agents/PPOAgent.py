import gym
import torch
from torch.distributions import Categorical

from project.agents.BaseAgent import BaseAgent
from project.environment.EnvWrapper import EnvWrapper
from project.models.policy_models import PolicyModelEncoder, PolicyModel


class PPOAgent(BaseAgent):
    def __init__(self, env: EnvWrapper, actor, critic, optimizer=None):
        self.env = env
        self.actor = actor
        self.critic = critic
        self.optimizer = optimizer

    def act(self, state):
        with torch.no_grad():

        pass

    def eval(self):
        with torch.no_grad():

        pass

    def train(self, num_episodes=5, num_steps=100):
        for i in range(num_episodes):

            s1 = self.env.reset()
            SAR = []

            for j in range(num_steps):
                s = s1

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
    seq_len = 10
    env_wrapper = EnvWrapper('procgen:procgen-starpilot-v0', seq_len)

    actor = PolicyModelEncoder(seq_len, 64, 64, env_wrapper.env.action_space.n)
    critic = PolicyModel(seq_len, 64, 64, 1)

    agent = PPOAgent(env_wrapper, actor, critic)
    agent.train()
