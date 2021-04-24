class MemBuffer:

    observations     = []
    actions          = []
    action_probs_old = []
    action_probs_new = []
    rewards          = []
    done             = []
    masks            = []
    ep_lengths       = []

    def clear(self):
        self.observations = []
        self.actions = []
        self.action_probs_old = []
        self.action_probs_new = []
        self.rewards = []
        self.done = []
        self.masks = []
        self.ep_lengths = []