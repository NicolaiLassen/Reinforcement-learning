import copy

import torch
from torch.distributions import Categorical

from project.agents.BaseAgent import BaseAgent
from project.environment.EnvWrapper import EnvWrapper
from project.models.policy_models import PolicyModelEncoder, PolicyModel
from utils.MemBuffer import MemBuffer


## TODO ENV BATCH FRAMES
class PPOAgent(BaseAgent):
    mem_buffer: MemBuffer = None

    def __init__(self, env: EnvWrapper, actor, critic, optimizer, action_space_n, accumulate_gradient=2,
                 gamma=0.9, eps_c=0.2,
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
        self.action_space_n = action_space_n
        self.n_max_times_update = n_max_times_update

    def train(self, max_time, max_time_steps):
        self.mem_buffer = MemBuffer(self.action_space_n, max_time)
        update_every = max_time * self.n_max_times_update
        t = 0
        while t < max_time_steps:
            s1 = self.env.reset()
            for ep_t in range(max_time + 1):
                t += 1
                s = s1
                action, log_probs = self.act(s)
                s1, r, d, _ = self.env.step(action)
                # TODO
                self.mem_buffer.set_next(s1, r, action, log_probs, d)

                # mask = []  # self.mem_buffer.masks.append(mask)
                if t % update_every == 0:
                    self.__update()

    def act(self, state):
        with torch.no_grad():
            state = state.unsqueeze(0).permute(1, 0, 2)
            action_logs_prob = self.actor_old(state)
        action_dist = Categorical(action_logs_prob)
        action = action_dist.sample()
        action_dist_log_prob = action_dist.log_prob(action)
        return action.detach().item(), action_dist_log_prob.detach()

    def evaluate(self, mem_states, mem_actions):
        mem_states = mem_states.unsqueeze(0).permute(1, 0, 2)
        action_prob = self.actor(mem_states)
        dist = Categorical(action_prob)
        action_log_probs = dist.log_prob(mem_actions)
        state_values = self.critic(mem_states)
        return action_log_probs, state_values

    def __calc_advantages(self, state_values):
        discounted_rewards = []
        running_reward = 0
        for r, d in zip(reversed(self.mem_buffer.rewards), reversed(self.mem_buffer.done)):
            if d:
                discounted_rewards.append(0)
                continue
            running_reward = r + (running_reward * self.gamma)
            discounted_rewards.append(running_reward)

        return torch.tensor(discounted_rewards, dtype=torch.float32) - state_values.detach()

    def __update(self):

        # ACC Gradient
        for _ in range(self.accumulate_gradient):
            log_probs, state_values = self.evaluate(self.mem_buffer.states, self.mem_buffer.actions)
            advantages = self.__calc_advantages(state_values)

            self.optimizer.zero_grad()
            loss = self.__calc_objective(log_probs, self.mem_buffer.action_log_probs, advantages).mean()
            print(loss)
            loss.backward()
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

    # USE CUDA
    agent = PPOAgent(env_wrapper, actor, critic, optimizer, env_wrapper.env.action_space.n)
    agent.train(400, 100000)
