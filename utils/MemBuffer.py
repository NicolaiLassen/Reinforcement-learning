class MemBuffer:

    observations = []
    actions      = []
    rewards      = []
    done         = []
    masks        = []
    ep_lengths   = []

    def clear(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.done = []