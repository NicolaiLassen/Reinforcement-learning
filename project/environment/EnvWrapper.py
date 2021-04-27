import os

import gym
import numpy as np
import torch
from torchvision import transforms

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


## TODO TAKE BATCH OF FRAMES 4-8
class EnvWrapper(gym.Env):
    def __init__(self, environment, seq_len=4, width: int = 64, height: int = 64, frameskip: int = 8, frames: int = 4):
        self.width = width
        self.height = height

        self.seq_len = seq_len
        self.env = gym.make(environment)
        self.env.frameskip = frameskip
        self.frames = frames

        self.transformer = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5)
        ])

    def step(self, action):
        observations = None
        info = None  # DO WE NEED THIS
        acc_reward = 0
        acc_done = False
        for i in range(self.frames): # HACK TO FOR NOW TODO
            obs, reward, done, info = self.env.step(action)
            if i == 0:
                observations = self.transformer(obs)
            else:
                observations = torch.cat([observations, self.transformer(obs)], dim=0)
            acc_reward += reward
            if done and (not acc_done):  # fast hack if it overlaps with done run # TODO
                acc_done = done
        # PLOT HOW THIS IS GOING
        # plt.imshow(obs)
        # plt.show()
        frames = observations.view(-1, self.width * self.height * self.frames)
        return frames, acc_reward, acc_done, info

    def reset(self):
        observations = torch.zeros((64, 64))
        obs = self.env.reset()
        for i in range(4):
            if i == 0:
                observations = self.transformer(obs)
            else:
                observations = torch.cat([observations, self.transformer(obs)])
        frame = observations.view(-1, self.width * self.height * self.frames)
        return frame

    def render(self, **kwargs):
        return self.env.render()
