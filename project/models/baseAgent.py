class BaseAgent:
    def act(self, act):
        raise NotImplementedError
    def eval(self):
        raise NotImplementedError
    def train(self):
        raise NotImplementedError