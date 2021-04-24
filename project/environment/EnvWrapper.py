import gym
import torch
from torchvision import transforms

from utils.MemBuffer import MemBuffer


class EnvWrapper(gym.Env):
    def __init__(self, environment, seq_len=4, width: int = 64, height: int = 64):

        self.width = width
        self.height = height

        self.seq_len = seq_len
        self.env = gym.make(environment)
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
        mask = torch.ones(self.seq_len, self.seq_len)

        for i in range(self.seq_len):
            obs, reward, done, info = self.env.step(action)
            frame = self.transformer(obs)
            frame = frame.squeeze().view(-1)
            seq[i] = frame
            buf_seq.append((frame, action, reward, done))
            if done:
                contains_done = True
                mask[i] = torch.zeros(self.seq_len)
                continue

        return seq, buf_seq, contains_done

    def reset(self):
        obs = self.env.reset()
        seq = self.__get_Buffer()
        frame = self.transformer(obs)
        frame = frame.squeeze().view(-1)
        for i in range(self.seq_len):
            seq[i] = frame

        mask = torch.zeros(self.seq_len, self.seq_len)
        mask[0] = torch.ones(self.seq_len)
        return seq, mask

    def render(self, **kwargs):
        return self.env.render()

    def __get_Buffer(self):
        return torch.zeros(self.seq_len, self.width * self.height)
