from torch.optim import Optimizer

from project.models import ActorCritic


class MemBuffer():
    def __init__(self, size=10000):
        super(MemBuffer, self).__init__()
        self.mem_size = size


class PPO():
    def __init__(self, optim: Optimizer, model: ActorCritic):
        super(PPO, self).__init__()

        self.optim = optim
        self.model = model

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
