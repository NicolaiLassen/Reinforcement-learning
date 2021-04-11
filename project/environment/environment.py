import gym
import torch
import torchvision
import numpy as np


class EnvWrapper:
    def __init__(self, environment, step_size=4):
        self.env = gym.make(environment)
        self.step_size = step_size

    def step(self, action):
        total_reward = 0
        frames = torch.Tensor(self.step_size, 64, 64)
        for i in range(self.step_size):
            obs, reward, done, _ = self.env.step(action)
            frame = self.transform(obs)
            frames[i] = frame
            total_reward += reward
            if done:
                break
        return frames, total_reward

    def transform(self, frame):
        frame = self._permute(frame)
        frame = self._to_greyscale(frame)
        return frame

    def _to_greyscale(self, frame):
        return torch.mean(torch.tensor(frame, dtype=torch.float), dim=0)

    def _permute(self, frame):
        frame = np.transpose(frame, (2, 0, 1))
        return frame

    def reset(self):
        return self.env.reset()