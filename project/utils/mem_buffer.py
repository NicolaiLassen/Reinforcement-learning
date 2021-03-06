import torch


class MemBuffer:
    t = 0

    def __init__(self, max_time, width=64, height=64, motion_blur=4, action_space_n=15):
        self.max_length = max_time
        self.width = width
        self.height = height
        self.motion_blur = motion_blur
        self.action_space_n = action_space_n

        self.states = torch.zeros(self.max_length, self.motion_blur, self.height, self.width, dtype=torch.float).cuda()
        self.next_states = torch.zeros(self.max_length, self.motion_blur, self.height, self.width,
                                       dtype=torch.float).cuda()
        self.rewards = torch.zeros(self.max_length, dtype=torch.float).cuda()
        self.actions = torch.zeros(self.max_length, dtype=torch.long).cuda()
        self.action_probs = torch.zeros(self.max_length, action_space_n, dtype=torch.float).cuda()
        self.action_log_prob = torch.zeros(self.max_length, dtype=torch.float).cuda()
        self.done = torch.zeros(self.max_length, dtype=torch.int)

    def set_next(self, state, next_state, reward, action, action_probs, action_log_prob, done, mask=None):
        if self.max_length == self.t:
            print("DON'T JUST TAKE ALL OF MY SPACE, YOU SON OF A GUN!")
            return
        self.states[self.t] = state
        self.next_states[self.t] = next_state
        self.rewards[self.t] = float(reward)
        self.actions[self.t] = action
        self.action_probs[self.t] = action_probs
        self.action_log_prob[self.t] = action_log_prob
        self.done[self.t] = int(done)
        self.t += 1

    def clear(self):
        self.t = 0
        self.states = torch.zeros(self.max_length, self.motion_blur, self.height, self.width, dtype=torch.float).cuda()
        self.next_states = torch.zeros(self.max_length, self.motion_blur, self.height, self.width,
                                       dtype=torch.float).cuda()
        self.rewards = torch.zeros(self.max_length, dtype=torch.float).cuda()
        self.actions = torch.zeros(self.max_length, dtype=torch.long).cuda()
        self.action_probs = torch.zeros(self.max_length, self.action_space_n, dtype=torch.float).cuda()
        self.action_log_prob = torch.zeros(self.max_length, dtype=torch.float).cuda()
        self.done = torch.zeros(self.max_length, dtype=torch.int)
