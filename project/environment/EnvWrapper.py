import gym
from torchvision import transforms


## TODO TAKE BATCH OF FRAMES 4-8
class EnvWrapper(gym.Env):
    def __init__(self, environment, seq_len=4, width: int = 64, height: int = 64, frameskip: int = 60):
        self.width = width
        self.height = height

        self.seq_len = seq_len
        self.env = gym.make(environment)
        self.env.frameskip = frameskip

        self.transformer = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5)
        ])

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        frame = self.transformer(obs).view(-1, self.width * self.height)
        return frame, reward, done, info

    def reset(self):
        obs = self.env.reset()
        frame = self.transformer(obs).view(-1, self.width * self.height)
        return frame

    def render(self, **kwargs):
        return self.env.render()
