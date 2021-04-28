import copy

import torch
import torch.nn as nn
import torch.optim as optim

from procgen import ProcgenGym3Env
from torch.distributions import Categorical

from project.agents.BaseAgent import BaseAgent
from project.environment.EnvWrapper import EnvWrapper
from project.models.policy_models import PolicyModelEncoder, PolicyModel
from utils.MemBuffer import MemBuffer


## TODO GPU MY MAN
## TODO ENV BATCH FRAMES
class PPOAgent(BaseAgent):
    mem_buffer: MemBuffer = None  # only used in traning

    def __init__(self,
                 env: EnvWrapper,
                 actor: nn.Module,
                 critic: nn.Module,
                 optimizer: optim.Optimizer,
                 n_acc_gradient=20,
                 gamma=0.9,
                 eps_c=0.2,
                 n_max_times_update=1):
        self.env = env
        self.actor = actor
        self.actor_old = copy.deepcopy(actor)
        self.actor_old.load_state_dict(actor.state_dict())
        self.critic = critic
        self.optimizer = optimizer
        self.action_space_n = env.env.action_space.n  # TODO
        # Hyper
        self.n_acc_gradient = n_acc_gradient
        self.gamma = gamma
        self.eps_c = eps_c
        self.n_max_times_update = n_max_times_update

    def train(self, max_time: int, max_time_steps: int):
        self.mem_buffer = MemBuffer(max_time)
        update_every = max_time * self.n_max_times_update  # TODO: THIS IS FOR THE BATCH PART
        t = 0
        while t < max_time_steps:
            s1 = self.env.reset()
            self.save_actor()
            for ep_t in range(max_time + 1):
                t += 1
                s = s1
                action, log_probs = self.act(s)
                s1, r, d, _ = self.env.step(action)
                self.mem_buffer.set_next(s, r, action, log_probs, d, self.mem_buffer.get_mask(d))
                if t % update_every == 0:
                    self.__update()

    def act(self, state):
        action_logs_prob = self.actor_old(state)
        action_dist = Categorical(action_logs_prob)
        action = action_dist.sample()
        action_dist_log_prob = action_dist.log_prob(action)
        return action.detach().item(), action_dist_log_prob.detach()

    def save_actor(self):
        print("save_actor")
        # torch.save(self.actor_old.state_dict(), "encoder_actor.ckpt")

    def load_actor(self, path):
        self.actor.load_state_dict(torch.load(path))
        self.actor_old.load_state_dict(torch.load(path))

    def __eval(self):
        action_prob = self.actor(self.mem_buffer.states, self.mem_buffer.masks)
        dist = Categorical(action_prob)
        action_log_probs = dist.log_prob(self.mem_buffer.actions)
        state_values = self.critic(self.mem_buffer.states)
        return action_log_probs, state_values

    def __calc_advantages(self, state_values):
        discounted_rewards = []
        running_reward = 0
        # Good old Monte, No fancy stuff
        for r, d in zip(reversed(self.mem_buffer.rewards), reversed(self.mem_buffer.done)):
            running_reward = r + (running_reward * self.gamma) * (1. - d)  # Zero out done states
            discounted_rewards.append(running_reward)

        return torch.tensor(discounted_rewards, dtype=torch.float32) - state_values.detach()

    def __update(self):
        # ACC Gradient traning
        # We have the samples why not train a bit on it?
        losses_ = torch.zeros(self.n_acc_gradient)  # SOME PRINT STUFF
        for _ in range(self.n_acc_gradient):
            self.optimizer.zero_grad()
            log_probs, state_values = self.__eval()
            advantages = self.__calc_advantages(state_values)
            loss = self.__calc_objective(log_probs, advantages)
            loss.backward()
            self.optimizer.step()

            # SOME PRINT STUFF
            with torch.no_grad():
                losses_[_] = loss.item()

        print("Mean ep losses: ", losses_.mean())
        print("Total ep reward: ", self.mem_buffer.rewards.sum())
        self.actor_old.load_state_dict(self.actor.state_dict())
        self.mem_buffer.clear()

    def __calc_objective(self, theta_log_probs, A_t):
        c_s_o = self.__clipped_surrogate_objective(theta_log_probs, A_t)
        return torch.mean(-c_s_o)

    def __clipped_surrogate_objective(self, theta_log_probs, A_t):
        r_t = torch.exp(theta_log_probs - self.mem_buffer.action_log_probs)
        r_t_c = torch.clamp(r_t, min=1 - self.eps_c, max=1 + self.eps_c)
        return torch.min(r_t * A_t, r_t_c * A_t)


if __name__ == "__main__":
    seq_len = 1
    bach_size = 1
    width = 64
    height = 64

    lr_actor = 0.0005
    lr_critic = 0.001

    # SWITCH THIS IN EQ BATCHES - NO cheating and getting good at only one thing
    env_wrapper = EnvWrapper('procgen:procgen-starpilot-v0', num_levels=1, difficulty="easy")

    actor = PolicyModelEncoder(seq_len, width, height, env_wrapper.env.action_space.n).cuda()
    critic = PolicyModel(seq_len, width, height).cuda()

    optimizer = torch.optim.Adam([
        {'params': actor.parameters(), 'lr': lr_actor},
        {'params': critic.parameters(), 'lr': lr_critic}
    ])

    agent = PPOAgent(env_wrapper, actor, critic, optimizer)
    agent.train(400, 100000)
