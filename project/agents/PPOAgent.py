import gym
import torch
from torch.distributions import Categorical

from project.agents.BaseAgent import BaseAgent
from project.environment.EnvWrapper import EnvWrapper
from project.models.policy_models import PolicyModelEncoder, PolicyModel


class PPOAgent(BaseAgent):
    def __init__(self, env: EnvWrapper, actor, critic, optimizer=None, gamma=0.9):
        self.env = env
        self.actor = actor
        self.critic = critic
        self.optimizer = optimizer
        self.gamma = gamma

    def act(self, state):
        pass

    def eval(self):
        pass

    def train(self, num_episodes=5, num_steps=1000):
        for i in range(num_episodes):

            s1 = self.env.reset()

            for j in range(num_steps):
                s = s1

                act_probs = self.actor(s)
                act_dist = Categorical(act_probs)
                act = act_dist.sample()

                frame_seq, buffer, done = self.env.step(act)
                if done:
                    s1 = self.env.reset()
                    continue

                discounted_rewards = self.calc_disc_rewards(buffer)

                r_net = self.critic(s)

    # Probably won't work - buffer doesn't contain whole episode, wrong return
    def calc_disc_rewards(self, buffer):

        disc_rewards = []
        running_rew = 0

        for _, _, reward, _ in reversed(buffer):
            disc_rewards.append(self.gamma*running_rew + reward)

        return disc_rewards

if __name__ == "__main__":
    seq_len = 4
    env_wrapper = EnvWrapper('procgen:procgen-starpilot-v0', seq_len)

    actor = PolicyModelEncoder(seq_len, 64, 64, env_wrapper.env.action_space.n)
    critic = PolicyModel(seq_len, 64, 64, 1)

    agent = PPOAgent(env_wrapper, actor, critic)
    agent.train()
