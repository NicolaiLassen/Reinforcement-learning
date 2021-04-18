from torch import nn
from torch.optim import Optimizer


class MemBuffer():
    def __init__(self, batch_size, size=10000, ):
        super(MemBuffer, self).__init__()
        self.mem_size = size
        self.mem = []
        self.batch_size = batch_size

    def __len__(self):
        return len(self.mem)

    def generate(self):
        pass

    def clear(self):
        pass

class PPO():
    def __init__(self,
                 actor: nn.Module,
                 critic: nn.Module,
                 optim: Optimizer,
                 batch_size=64,
                 curiosity=0.1
                 ):
        super(PPO, self).__init__()

        self.optim = optim
        self.actor = actor
        self.critic = critic
        self.memBuffer = MemBuffer(batch_size)

    ## SEE PPO to use both an actor an critic network

    def act(self, state):
        self.model()
        ## PPO

        # return action, action_logprob

    def evaluate(self, state, action):
        ## PPO
        self.model()

        # return action_logprobs, state_values, dist_prob

    def train(self):
        self.optim.zero_grad()

        self.actor.backward()

        self.optim.step()
