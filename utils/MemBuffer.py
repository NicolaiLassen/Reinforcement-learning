import torch


## class 2DBatchMemBuffer

class MemBuffer:
    ## Current single ep MemBuffer
    t = 0

    ## Should take all probs not just from dist
    # Pass in action values
    def __init__(self, max_time, width=64, height=64):
        self.max_length = max_time
        self.width = width
        self.height = height

        self.states = torch.zeros(self.max_length, self.width * self.height, dtype=torch.float)
        self.rewards = torch.zeros(self.max_length, dtype=torch.float)
        self.actions = torch.zeros(self.max_length, dtype=torch.int)
        self.action_log_probs = torch.zeros(self.max_length, dtype=torch.float)
        self.done = torch.zeros(self.max_length, dtype=torch.int)
        self.masks = torch.zeros(self.max_length, self.max_length)

    def set_next(self, state, reward, action, action_log_prob, done, mask=None):
        if self.max_length == self.t:
            print("DON'T JUST TAKE ALL OF MY SPACE, YOU SON OF A GUN!")
            return
        self.states[self.t] = state
        self.rewards[self.t] = float(reward)
        self.actions[self.t] = action
        self.action_log_probs[self.t] = action_log_prob
        self.done[self.t] = int(done)
        if mask is not None:
            self.masks[self.t] = mask
        self.t += 1

    def clear(self):
        self.t = 0
        self.states = torch.zeros(self.max_length, self.width * self.height, dtype=torch.float)
        self.rewards = torch.zeros(self.max_length, dtype=torch.float)
        self.actions = torch.zeros(self.max_length, dtype=torch.int)
        self.action_log_probs = torch.zeros(self.max_length, dtype=torch.float)
        self.done = torch.zeros(self.max_length, dtype=torch.int)
        self.masks = torch.zeros(self.max_length, self.max_length)

    def get_mask(self, skip=False):
        # Very simple mask
        # We don't wan't the encoder to learn shit! Speed up that learning!
        return torch.zeros(self.max_length) if skip else torch.ones(self.max_length)
