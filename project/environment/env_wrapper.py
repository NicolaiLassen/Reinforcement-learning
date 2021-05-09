import os

import gym
import torch
from torchvision import transforms

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


## TODO TAKE BATCH OF FRAMES 4-8
class EnvWrapper(gym.Env):
    def __init__(self, environment,
                 num_levels=200,  # limted time hard limit 24 hour
                 # https://www.aicrowd.com/challenges/neurips-2020-procgen-competition
                 # Procgen environments require training on 500–1000 different levels before they can generalize to new levels
                 difficulty='easy',  # Let's train on easy for testing
                 start_level=0,
                 width: int = 64,
                 height: int = 64,
                 frameskip: int = 2,
                 motion_blur: int = 4):
        self.width = width
        self.height = height

        self.env = gym.make(environment, start_level=start_level, num_levels=num_levels, distribution_mode=difficulty)
        self.env.frameskip = frameskip
        self.motion_blur = motion_blur

        self.obs_transformer = transforms.Compose([
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
        for i in range(self.motion_blur):
            obs, reward, done, info = self.env.step(action)
            if i == 0:
                observations = self.obs_transformer(obs)
            else:
                observations = torch.cat([observations, self.obs_transformer(obs)], dim=0)
            acc_reward += reward
            if done and (not acc_done):
                acc_done = done

        return observations.cuda(), acc_reward, acc_done, info

    def reset(self):
        observations = torch.zeros((64, 64))
        obs = self.env.reset()
        for i in range(4):
            if i == 0:
                observations = self.obs_transformer(obs)
            else:
                observations = torch.cat([observations, self.obs_transformer(obs)])
        return observations.cuda()

    def render(self, **kwargs):
        return self.env.render()
