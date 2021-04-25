# Agent Base Interface
class BaseAgent:
    def act(self, act):
        raise NotImplementedError

    def eval(self):
        raise NotImplementedError

    def train(self, max_time, max_time_steps):
        raise NotImplementedError
