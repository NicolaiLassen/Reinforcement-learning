from torch.optim import Optimizer
from torch.nn import Module

class Buffer():



class PPO():
    def __init__(self, optim: Optimizer, model: Module ):
        super(PPO, self).__init__()

        self.optim = optim
        self.model = model

    def train(self):
        self.optim.zero_grad()

        self.model.backward()

        self.optim.step()


