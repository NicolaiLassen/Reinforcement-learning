class MemBuffer():
    def __init__(self, batch_size, size=10000, ):
        super(MemBuffer, self).__init__()
        self.mem_size = size
        self.mem = []
        self.batch_size = batch_size

    def add(self, observation, action, reward, done):
        self.mem.append((observation, action, reward, done))
        if len(self.mem) > self.mem_size:
            self.mem = self.mem[-self.mem_size:]

    def __len__(self):
        return len(self.mem)

    def generate(self):
        pass

    def clear(self):
        pass
