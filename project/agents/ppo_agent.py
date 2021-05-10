import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from agents.base_agent import BaseAgent
from environment.env_wrapper import EnvWrapper
from utils.mem_buffer import MemBuffer
from utils.normalize_dist import normalize_dist


class PPOAgent(BaseAgent):
    mem_buffer: MemBuffer = None  # only used in traning

    # counters ckpt
    t_update = 0  # t * 1000
    model_save_every = 50  # (8000000/4)  / 2000 / 50

    intrinsic_reward_ckpt = []
    curiosity_loss_ckpt = []
    actor_loss_ckpt = []
    critic_loss_ckpt = []
    reward_ckpt = []

    def __init__(self,
                 env: EnvWrapper,
                 actor: nn.Module,
                 critic: nn.Module = None,
                 curiosity: nn.Module = None,
                 optimizer: optim.Optimizer = None,
                 name="vit",  # vit or conv
                 gamma=0.9,
                 eta=0.5,
                 beta=0.8,
                 eps_c=0.2,
                 loss_entropy_c=0.01,
                 n_max_Times_update=1):
        self.env = env
        self.actor = actor
        self.actor_old = copy.deepcopy(actor)
        self.critic = critic

        self.name = name
        self.optimizer = optimizer

        self.action_space_n = env.env.action_space.n
        # Curiosity
        self.curiosity = curiosity
        # Hyper n
        self.n_max_Times_update = n_max_Times_update
        # Hyper c
        self.gamma = gamma
        self.eta = eta
        self.beta = beta

        self.eps_c = eps_c
        self.loss_entropy_c = loss_entropy_c

    def train(self, max_time_per_batch: int, max_time_steps: int):
        self.mem_buffer = MemBuffer(max_time_per_batch, action_space_n=self.action_space_n)
        update_every = max_time_per_batch * self.n_max_Times_update  # TODO: BATCH
        t = 0
        s1 = self.env.reset()
        while t < max_time_steps:
            for ep_T in range(max_time_per_batch + 1):
                t += 1
                s = s1
                action, probs, log_prob = self.act(s)
                s1, r, d, _ = self.env.step(action)
                self.mem_buffer.set_next(s, s1, r, action, probs, log_prob, d)
                if t % update_every == 0:
                    self.__update()

    def act(self, state):
        action_logs_prob = self.actor_old(state.unsqueeze(0))
        action_dist = Categorical(action_logs_prob)
        action = action_dist.sample()
        action_dist_log_prob = action_dist.log_prob(action)
        return action.detach().item(), action_dist.probs.detach(), action_dist_log_prob.detach()

    def save_ckpt(self):

        base = "ckpt_ppo_{}/starpilot_easy".format(self.name)

        if self.t_update % self.model_save_every == 0:
            torch.save(self.actor_old.state_dict(),
                       "{}/actor_{}.ckpt".format(base, self.t_update))

        # torch.save(torch.tensor(self.curiosity_loss_ckpt),
        #            "{}/losses_curiosity.ckpt".format(base, self.name))
        #
        # torch.save(torch.tensor(self.intrinsic_reward_ckpt),
        #            "{}/intrinsic_rewards.ckpt".format(base, self.name))

        torch.save(torch.tensor(self.actor_loss_ckpt),
                   "{}/losses_actor.ckpt".format(base, self.name))

        torch.save(torch.tensor(self.critic_loss_ckpt),
                   "{}/losses_critic.ckpt".format(base, self.name))

        torch.save(torch.tensor(self.reward_ckpt),
                   "{}/rewards.ckpt".format(base, self.name))

        self.t_update += 1

    def load_actor(self, path):
        self.actor.load_state_dict(torch.load(path))
        self.actor_old.load_state_dict(torch.load(path))

    def __update(self):

        # defrag GPU Mem
        torch.cuda.empty_cache()

        action_log_probs, state_values, entropy = self.__eval()
        d_r = self.__discounted_rewards()
        A_T = normalize_dist(self.__advantages(d_r, state_values))

        # print(d_r)
        # print(A_T)
        # print(state_values)

        # r_i_ts, r_i_ts_loss, a_t_hat_loss = self.__intrinsic_reward_objective()
        R_T = A_T  # + r_i_ts

        actor_loss = - self.__clipped_surrogate_objective(action_log_probs, R_T)  # L^CLIP

        critic_loss = (0.5 * torch.pow(state_values - self.mem_buffer.rewards, 2)).mean()  # E # c1 L^VF
        # print(critic_loss)

        entropy_bonus = entropy * self.loss_entropy_c  # c2 S[]

        # curiosity_loss = (1 - (a_t_hat_loss * self.beta) + (r_i_ts_loss * self.beta))

        self.optimizer.zero_grad()
        # Gradient ascent -(actor_loss - critic_loss + entropy_bonus)
        # curiosity acent -(E phi()) + ICM_loss
        total_loss = (actor_loss + critic_loss - entropy_bonus).mean()
        # total_loss = (actor_loss - critic_loss + entropy_bonus).mean() + curiosity_loss
        total_loss.backward()
        self.optimizer.step()

        # Save CKPT
        with torch.no_grad():
            # self.intrinsic_reward_ckpt.append(intrinsic_rewards.sum().item())
            # self.curiosity_loss_ckpt.append(curiosity_losses.sum().item())
            self.actor_loss_ckpt.append(actor_loss.sum().item())
            self.critic_loss_ckpt.append(critic_loss.sum().item())
            self.reward_ckpt.append(self.mem_buffer.rewards.sum().item())
        self.save_ckpt()

        self.actor_old.load_state_dict(self.actor.state_dict())
        self.mem_buffer.clear()

    def __eval(self):
        action_logs_prob = self.actor(self.mem_buffer.states)
        dist = Categorical(action_logs_prob)
        action_log_prob = dist.log_prob(self.mem_buffer.actions)
        state_values = self.critic(self.mem_buffer.states.unsqueeze(0))
        return action_log_prob, state_values.squeeze(1), dist.entropy()

    def __advantages(self, discounted_rewards, state_values):
        advantages = torch.zeros(len(discounted_rewards))
        T = self.mem_buffer.max_length - 1
        last_state_value = state_values[T]
        t = 0
        for discounted_reward in discounted_rewards:
            advantages[t] = discounted_reward - state_values[t] + last_state_value * (
                    self.gamma ** (T - t))
            t += 1
        return advantages.float().cuda().detach()

    def __discounted_rewards(self):
        discounted_rewards = torch.zeros(len(self.mem_buffer.rewards))
        running_reward = 0
        t = 0
        for r, d in zip(reversed(self.mem_buffer.rewards), reversed(self.mem_buffer.done)):
            running_reward = r + (running_reward * self.gamma) * (1. - d)  # Zero out done states
            discounted_rewards[-t] = running_reward
            t += 1
        return discounted_rewards.float().cuda()

    def __intrinsic_reward_objective(self):
        next_states = self.mem_buffer.next_states
        states = self.mem_buffer.states
        action_probs = self.mem_buffer.action_probs
        actions = self.mem_buffer.actions

        a_t_hats, phi_t1_hats, phi_t1s, phi_ts = self.curiosity(states, next_states, action_probs)
        r_i_ts = F.mse_loss(phi_t1_hats, phi_t1s, reduction='none').sum(-1)

        return (self.eta * r_i_ts).detach(), r_i_ts.mean(), F.cross_entropy(a_t_hats, actions)

    def __clipped_surrogate_objective(self, action_log_probs, A_T):
        # torch.exp(action_log_probs) / torch.exp(self.mem_buffer.action_log_prob)
        r_T_theta = torch.exp(action_log_probs - self.mem_buffer.action_log_prob)
        r_T_c_theta = torch.clamp(r_T_theta, min=1 - self.eps_c, max=1 + self.eps_c)
        return torch.min(r_T_theta * A_T, r_T_c_theta * A_T).mean()  # E
