import gym
import torch
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

    # Take 'step_size' steps with the same action
    def step(self, action):

        seq = torch.zeros(self.step_size, 64, 64)

        for i in range(self.step_size):
            obs, reward, done, info = self.env.step(action)
            frame = self.transformer(obs)
            frame = frame.squeeze()
            seq[i] = frame

        return seq

    def reset(self):
        obs = self.env.reset()
        return self.transformer(obs)
