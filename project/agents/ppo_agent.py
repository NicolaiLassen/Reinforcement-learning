import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from project.agents.base_agent import BaseAgent
from project.environment.env_wrapper import EnvWrapper
from project.models.policy_models import PolicyModelEncoder, PolicyModel
from utils.curiosity import ICM
from utils.mem_buffer import MemBuffer
from utils.normalize_dist import normalize_dist


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
                 n_max_Times_update=1):
        self.env = env
        self.actor = actor
        self.actor_old = copy.deepcopy(actor)
        self.actor_old.load_state_dict(actor.state_dict())
        self.critic = critic
        self.optimizer = optimizer
        self.action_space_n = env.env.action_space.n
        # Curiosity
        self.ICM = ICM(self.action_space_n).cuda()
        # Hyper n
        self.n_acc_grad = n_acc_gradient
        self.n_max_Times_update = n_max_Times_update
        # Hyper c
        self.gamma = gamma
        self.eps_c = eps_c
        self.loss_entropy_c = 0.01
        self.intrinsic_curiosity_c = 0.9
        self.eta = 0.5

    def train(self, max_Time: int, max_Time_steps: int):
        self.mem_buffer = MemBuffer(max_Time)
        update_every = max_Time * self.n_max_Times_update  # TODO: BATCH
        t = 0
        s1 = self.env.reset()
        while t < max_Time_steps:
            self.save_actor()
            for ep_T in range(max_Time + 1):
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
        return action.detach().item(), action_dist.probs

    def save_actor(self):
        print("save_actor")
        # torch.save(self.actor_old.state_dict(), "encoder_actor.ckpt")

    def load_actor(self, path):
        self.actor.load_state_dict(torch.load(path))
        self.actor_old.load_state_dict(torch.load(path))

    def __update(self):
        losses_ = torch.zeros(self.n_acc_grad)  # SOME PRINT STUFF
        # ACC Gradient traning
        # We have the samples why not train a bit on it?
        for _ in range(self.n_acc_grad):
            self.optimizer.zero_grad()
            action_log_probs, state_values, entropy = self.__eval()

            A_T = self.__advantages(state_values)
            r_i_ts, r_i_ts_loss, a_t_hat_loss = self.__intrinsic_reward_objective()
            R_T = normalize_dist(A_T + (r_i_ts * self.intrinsic_curiosity_c))

            c_s_o = self.__clipped_surrogate_objective(action_log_probs, R_T)

            curiosity_loss = r_i_ts_loss + a_t_hat_loss
            print(curiosity_loss)
            c_s_o_loss = (-c_s_o - (entropy * self.loss_entropy_c)).mean()

            loss = c_s_o_loss + curiosity_loss
            loss.backward()

            self.optimizer.step()

            # SOME PRINT STUFF
            with torch.no_grad():
                losses_[_] = loss.item()

        print("Mean ep losses: ", losses_.mean())
        print("Total ep reward: ", self.mem_buffer.rewards.sum())
        self.actor_old.load_state_dict(self.actor.state_dict())
        self.mem_buffer.clear()

    def __eval(self):
        action_prob = self.actor(self.mem_buffer.states)
        action_dist = Categorical(action_prob)
        state_values = self.critic(self.mem_buffer.states)
        return action_dist.probs, state_values, action_dist.entropy()  # Bregman divergence

    def __advantages(self, state_values):
        discounted_rewards = []
        running_reward = 0

        for r, d in zip(reversed(self.mem_buffer.rewards), reversed(self.mem_buffer.done)):
            running_reward = r + (running_reward * self.gamma) * (1. - d)  # Zero out done states
            discounted_rewards.append(running_reward)

        return torch.tensor(discounted_rewards, dtype=torch.float32).cuda() - state_values.detach()

    def __intrinsic_reward_objective(self):
        next_states = self.mem_buffer.states[1:]
        states = self.mem_buffer.states[:-1]
        action_probs = self.mem_buffer.action_log_probs[:-1]
        actions = self.mem_buffer.actions[:-1]

        a_t_hats, phi_t1_hats, phi_t1s, phi_ts = self.ICM(states, next_states, action_probs)

        r_i_ts = F.mse_loss(phi_t1_hats, phi_t1s, reduce=False).sum(-1).mean()

        return (self.eta * r_i_ts).detach(), r_i_ts, F.cross_entropy(a_t_hats, actions)

    def __clipped_surrogate_objective(self, action_log_probs, R_T):
        r_T_theta = torch.exp(action_log_probs - self.mem_buffer.action_log_probs)
        r_T_c_theta = torch.clamp(r_T_theta, min=1 - self.eps_c, max=1 + self.eps_c)
        return torch.min(r_T_theta * R_T, r_T_c_theta * R_T).mean()  # E


if __name__ == "__main__":
    bach_size = 1
    width = 64
    height = 64

    nn.Linear(20, 30)

    lr_actor = 0.0005
    lr_critic = 0.001

    # SWITCH THIS IN EQ BATCHES - NO cheating and getting good at only one thing
    env_wrapper = EnvWrapper('procgen:procgen-starpilot-v0', num_levels=1, difficulty="easy")

    actor = PolicyModelEncoder(width, height, env_wrapper.env.action_space.n).cuda()
    critic = PolicyModel(width, height).cuda()

    optimizer = torch.optim.Adam([
        {'params': actor.parameters(), 'lr': lr_actor},
        {'params': critic.parameters(), 'lr': lr_critic}
    ])

    agent = PPOAgent(env_wrapper, actor, critic, optimizer)
    agent.train(10, 100000)
