import gym
import torch
from torchvision import transforms

from utils.MemBuffer import MemBuffer


class EnvWrapper(gym.Env):
    def __init__(self, environment, step_size=4, width: int = 64, height: int = 64):

        self.width = width
        self.height = height

        self.step_size = step_size
        self.env = gym.make(environment)
        self.mem_buffer = MemBuffer(batch_size=100)
        self.transformer = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5)
        ])

    def step(self, action):
        seq = self.__get_Buffer()
        buf_seq = []
        contains_done = False

        for i in range(self.step_size):
            obs, reward, done, info = self.env.step(action)
            frame = self.transformer(obs)
            frame = frame.squeeze()
            seq[i] = frame
            buf_seq.append((frame, action, reward, done))
            if done:
                contains_done = True
                continue

        return seq, buf_seq, contains_done

    def reset(self):
        obs = self.env.reset()
        seq = self.__get_Buffer()
        frame = self.transformer(obs)
        frame = frame.squeeze()
        for i in range(self.step_size):
            seq[i] = frame
        return seq

    def __get_Buffer(self):
        return torch.zeros(self.step_size, self.width, self.height)
