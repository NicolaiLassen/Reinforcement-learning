class MemBuffer:

    observations     = []
    actions          = []
    action_probs_old = []
    rewards          = []
    done             = []
    masks            = []
    ep_lengths       = []

    def clear(self):
        self.observations = []
        self.actions = []
        self.action_probs_old = []
        self.rewards = []
        self.done = []
        self.masks = []
        self.ep_lengths = []