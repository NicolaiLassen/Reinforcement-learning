class MemBuffer:

    observations = []
    actions      = []
    rewards      = []
    done         = []
    masks        = []

    def clear(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.done = []