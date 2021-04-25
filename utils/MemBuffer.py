class MemBuffer:

    def __init__(self):
        self.states = []
        self.rewards = []
        self.actions = []
        self.action_log_prob = []
        self.done = []
        self.ep_lengths = []

        self.masks = []

    def clear(self):
        self.states = []
        self.actions = []
        self.action_log_prob = []
        self.rewards = []
        self.done = []
        self.ep_lengths = []

        self.masks = []
