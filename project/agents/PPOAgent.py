import copy

import torch
from torch.autograd import Variable
from torch.distributions import Categorical

from project.agents.BaseAgent import BaseAgent
from project.environment.EnvWrapper import EnvWrapper
from project.models.policy_models import PolicyModelEncoder, PolicyModel
from utils.MemBuffer import MemBuffer


## TODO ENV BATCH FRAMES
class PPOAgent(BaseAgent):
    mem_buffer = MemBuffer()

    def __init__(self, env: EnvWrapper, actor, critic, optimizer, accumulate_gradient=2, gamma=0.9, eps_c=0.2,
                 n_max_times_update=1):
        self.env = env
        self.actor = actor
        self.actor_old = copy.deepcopy(actor)
        self.actor_old.load_state_dict(actor.state_dict())
        self.critic = critic
        self.optimizer = optimizer
        self.accumulate_gradient = accumulate_gradient
        self.gamma = gamma
        self.eps_c = eps_c
        self.n_max_times_update = n_max_times_update

    def train(self, max_time, max_time_steps):
        update_every = max_time * self.n_max_times_update
        t = 0
        while t < max_time_steps:
            s1 = self.env.reset()
            for ep_t in range(max_time + 1):
                t += 1
                s = s1
                selected_action = self.act(s)
                s1, r, d, _ = self.env.step(selected_action)
                self.mem_buffer.rewards.append(r)
                self.mem_buffer.done.append(d)
                # mask = []  # self.mem_buffer.masks.append(mask)
                if t % update_every == 0:
                    self.__update()

    def act(self, state_t):
        with torch.no_grad():
            state_t = state_t.unsqueeze(0).permute(1, 0, 2)
            action_logs_prob = self.actor_old(state_t)
        action_dist = Categorical(action_logs_prob)
        action = action_dist.sample()
        action_dist_log_prob = action_dist.log_prob(action)
        action = action.detach()
        action_dist_log_prob = action_dist_log_prob.detach()

        self.mem_buffer.states.append(state_t)
        self.mem_buffer.actions.append(action)
        self.mem_buffer.action_log_prob.append(action_dist_log_prob)

        return action.item()

    def evaluate(self, mem_states, mem_actions):
        mem_states = mem_states.permute(1, 0, 2)
        action_prob = self.actor(mem_states)
        dist = Categorical(action_prob)
        action_log_prob = dist.log_prob(mem_actions)
        state_values = self.critic(mem_states)
        return action_log_prob, state_values

    def __calc_advantages(self, state_values):
        discounted_rewards = []
        running_reward = 0
        for r, d in zip(reversed(self.mem_buffer.rewards), reversed(self.mem_buffer.done)):
            if d:
                discounted_rewards.append(0)
                continue
            running_reward = r + (running_reward * self.gamma)
            discounted_rewards.append(running_reward)

        # eval state value
        return torch.tensor(discounted_rewards, dtype=torch.float32) - state_values.detach()

    def __update(self):

        # FIX GRAD
        print(self.mem_buffer.rewards)
        mem_states = Variable(torch.stack(self.mem_buffer.states, dim=0).squeeze(1), requires_grad=True)
        mem_log_prob = Variable(torch.stack(self.mem_buffer.action_log_prob, dim=0), requires_grad=True)
        mem_actions = torch.stack(self.mem_buffer.actions, dim=0)
        # print(mem_states.shape)
        # print(mem_actions.shape)
        # print(mem_log_prob.shape)

        # ACC Gradient
        for _ in range(self.accumulate_gradient):
            log_prob, state_values = self.evaluate(mem_states, mem_actions)
            advantages = self.__calc_advantages(state_values)

            self.optimizer.zero_grad()
            loss = self.__calc_objective(log_prob, mem_log_prob, advantages)
            loss.mean().backward()
            self.optimizer.step()

        self.actor_old.load_state_dict(self.actor.state_dict())
        self.mem_buffer.clear()

    def __calc_objective(self, theta_log_probs, theta_log_probs_old, A_t):
        r_t = torch.exp(theta_log_probs - theta_log_probs_old)
        r_t_c = torch.clamp(r_t, min=1 - self.eps_c, max=1 + self.eps_c)
        return -torch.min(r_t * A_t, r_t_c * A_t)


if __name__ == "__main__":
    seq_len = 1
    bach_size = 1
    width = 64
    height = 64

    lr_actor = 0.0003
    lr_critic = 0.001

    env_wrapper = EnvWrapper('procgen:procgen-starpilot-v0', seq_len)

    actor = PolicyModelEncoder(seq_len, width, height, env_wrapper.env.action_space.n)
    critic = PolicyModel(seq_len, width, height)

    optimizer = torch.optim.Adam([
        {'params': actor.parameters(), 'lr': lr_actor},
        {'params': critic.parameters(), 'lr': lr_critic}
    ])

    agent = PPOAgent(env_wrapper, actor, critic, optimizer)
    agent.train(4000, 100000)
