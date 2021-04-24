import copy
import torch
from torch.distributions import Categorical

from project.agents.BaseAgent import BaseAgent
from project.environment.EnvWrapper import EnvWrapper
from project.models.policy_models import PolicyModelEncoder, PolicyModel
from utils.MemBuffer import MemBuffer


class PPOAgent(BaseAgent):
    mem_buffer = MemBuffer()

    def __init__(self, env: EnvWrapper, actor, critic, optimizer=None, gamma=0.9, eps_c=0.2):
        self.env = env
        self.actor_old = actor
        self.actor = copy.deepcopy(actor)
        self.critic = critic
        self.optimizer = optimizer
        self.gamma = gamma
        self.eps_c = eps_c

    def act(self, state):
        with torch.no_grad():
            act_probs = self.actor_old(state)
            act_dist = Categorical(act_probs)
            act = act_dist.sample()
            return act, act_probs

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

    def train(self, max_time, max_time_steps):

        s1, mask = self.env.reset()
        t = 0

        while t < max_time_steps:

            s1 = self.env.reset()

            for ep_t in range(max_time):
                t += 1
                s = s1
                ## (S, N, E)
                ## TEMP BATCH SIZE
                s_enc = s.unsqueeze(0).permute(1, 0, 2)
                act, act_probs = self.act(s_enc)
                s1, r, d, _ = self.env.step(act)

                #if update_check:
                    #TODO: update model

                self.mem_buffer.rewards.append(r)
                self.mem_buffer.done.append(d)
                self.mem_buffer.actions.append(act)
                self.mem_buffer.action_probs.append(act_probs)
                self.mem_buffer.observations.append(s_enc)
                if d:
                    break

    def calc_advantages(self):

        gamma = self.gamma
        discounted_rewards = []
        running_reward = 0

        for r in reversed(self.mem_buffer.rewards):
            running_reward += r + running_reward*gamma
            gamma *= gamma
            discounted_rewards.append(running_reward)

        state_values = get_state_values()

        return [(d - sv) for d, sv in zip(discounted_rewards, state_values)]

    def get_state_values(self):

        return []

    def calc_objective(self, probs, probs_old, A_t):
        # r_t = torch.div(probs, probs_batch) # Paper
        # https://cs.stackexchange.com/questions/70518/why-do-we-use-the-log-in-gradient-based-reinforcement-algorithms
        # Better gradient than div
        r_t = torch.exp(probs - probs_old)
        r_t_c = torch.clamp(r_t, min=1 - self.eps_c, max=1 + self.eps_c)
        return torch.min(r_t * A_t, r_t_c * A_t)

    def update_models(self):
        self.optimizer.zero_grad()
        advantages = self.calc_advantages()

if __name__ == "__main__":

    seq_len = 100
    bach_size = 1
    env_wrapper = EnvWrapper('procgen:procgen-starpilot-v0', seq_len)

    actor = PolicyModelEncoder(seq_len, 64, 64, env_wrapper.env.action_space.n)
    critic = PolicyModel(seq_len, 64, 64, 1)

    agent = PPOAgent(env_wrapper, actor, critic)
    agent.train()
