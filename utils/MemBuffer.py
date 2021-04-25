import torch


class MemBuffer:
    ## Current single ep MemBuffer
    t = 0

    def __init__(self, action_dim, max_time=400, width=64, height=64):
        self.action_dim = action_dim
        self.max_length = max_time
        self.width = width
        self.height = height

        self.states = torch.zeros(self.max_length, self.width * self.height, dtype=torch.float)
        self.rewards = torch.zeros(self.max_length, dtype=torch.float)
        self.actions = torch.zeros(self.max_length, dtype=torch.int)
        self.action_log_probs = torch.zeros(self.max_length, dtype=torch.float)
        self.done = torch.zeros(self.max_length, dtype=torch.bool)
        self.masks = torch.zeros(self.max_length, self.width * self.height)

    def set_next(self, state, reward, action, action_log_prob, done):
        if self.max_length == self.t:
            print("Space")
            return
        self.states[self.t] = state
        self.rewards[self.t] = float(reward)
        self.actions[self.t] = action
        self.action_log_probs[self.t] = action_log_prob
        self.done[self.t] = bool(done)
        self.t += 1

    def clear(self):
        self.t = 0
        self.states = torch.zeros(self.max_length, self.width * self.height, dtype=torch.float)
        self.rewards = torch.zeros(self.max_length, dtype=torch.float)
        self.actions = torch.zeros(self.max_length, dtype=torch.int)
        self.action_log_probs = torch.zeros(self.max_length, dtype=torch.float)
        self.done = torch.zeros(self.max_length, dtype=torch.bool)
        self.masks = torch.zeros(self.max_length, self.width * self.height)
