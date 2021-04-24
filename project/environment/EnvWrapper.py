import gym
from torchvision import transforms


class EnvWrapper(gym.Env):
    def __init__(self, environment, step_size=4):
        self.step_size = step_size
        self.env = gym.make(environment)
        self.transformer = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5)
        ])

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        frame = self.transformer(obs)
        return frame, reward, done, info

    def reset(self):
        return self.env.reset()
