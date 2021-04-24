import gym
import torch
from torchvision import transforms

from utils.MemBuffer import MemBuffer


class EnvWrapper(gym.Env):
    def __init__(self, environment, step_size=4):
        self.step_size = step_size
        self.env = gym.make(environment)
        self.mem_buffer = MemBuffer(batch_size=self.step_size)
        self.transformer = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5)
        ])

    # Take 'step_size' steps with the same action
    def step(self, action):

        seq = torch.zeros(self.step_size, 64, 64)
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
        return self.transformer(obs)
