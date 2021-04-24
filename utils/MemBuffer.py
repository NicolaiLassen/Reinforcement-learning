class MemBuffer:

    observations = []
    actions      = []
    rewards      = []
    done         = []

    def add(self, observation, action, reward, done):
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.done.append(done)
