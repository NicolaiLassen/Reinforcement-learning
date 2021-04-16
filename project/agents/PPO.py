from torch import nn
from torch.optim import Optimizer


class MemBuffer():
    def __init__(self, size=10000):
        super(MemBuffer, self).__init__()
        self.mem_size = size


class PPO():
    def __init__(self, actor: nn.Module, critic: nn.Module, optim: Optimizer):
        super(PPO, self).__init__()

        self.optim = optim
        self.actor = actor

    ## SEE PPO to use both an actor an critic network

    def act(self, state):
        self.model()
        ## PPO

        return action, action_logprob

    def evaluate(self, state, action):
        ## PPO
        self.model()

        return action_logprobs, state_values, dist_prob

    def train(self):
        self.optim.zero_grad()

        self.model.backward()

        self.optim.step()
