import copy

import torch
from torch.distributions import Categorical

from project.agents.BaseAgent import BaseAgent
from project.environment.EnvWrapper import EnvWrapper
from project.models.policy_models import PolicyModelEncoder, PolicyModel
from utils.MemBuffer import MemBuffer


class PPOAgent(BaseAgent):

    mem_buffer = MemBuffer()

    def __init__(self, env: EnvWrapper, actor, critic, optimizer=None, gamma=0.9):
        self.env = env
        self.actor_old = actor
        self.actor = copy.deepcopy(actor)
        self.critic = critic
        self.optimizer = optimizer
        self.gamma = gamma

    def act(self, state):
        with torch.no_grad():
            act_probs = self.actor_old(state)
            act_dist = Categorical(act_probs)
            act = act_dist.sample()
            self.mem_buffer.observations.append(state)
            self.mem_buffer.actions.append(act)
            return act


    def eval(self):
        pass

    def train(self, num_episodes=5, num_steps=100):
        for i in range(num_episodes):

            s1, mask = self.env.reset()

            for j in range(num_steps):
                s = s1
                ## (S, N, E)
                ## TEMP BATCH SIZE
                s_enc = s.unsqueeze(0).permute(1, 0, 2)
                print(s_enc.shape)
                print(mask)
                act_probs = self.actor(s_enc, mask)
                act_dist = Categorical(act_probs)
                act = act_dist.sample()
                frame_seq, buffer, done = self.env.step(act)
                if done:
                    s1 = self.env.reset()
                    continue
                r_net = self.critic(s)


    def rollout(self, timesteps):

        s1, mask = self.env.reset()
        advantages = []
        self.mem_buffer.clear()

        for t in timesteps:
            s = s1
            ## (S, N, E)
            ## TEMP BATCH SIZE
            s_enc = s.unsqueeze(0).permute(1, 0, 2)
            act = self.act(s_enc)
            s1, r, d, _ = self.env.step(act)
            self.mem_buffer.rewards.append(r)
            self.mem_buffer.done.append(d)
            if d:
                s1, mask = self.env.reset()
                continue
        advantages = self.calc_advantages()

    def calc_advantages(self):





if __name__ == "__main__":
    seq_len = 100
    bach_size = 1
    env_wrapper = EnvWrapper('procgen:procgen-starpilot-v0', seq_len)

    actor = PolicyModelEncoder(seq_len, 64, 64, env_wrapper.env.action_space.n)
    critic = PolicyModel(seq_len, 64, 64, 1)

    agent = PPOAgent(env_wrapper, actor, critic)
    agent.train()
