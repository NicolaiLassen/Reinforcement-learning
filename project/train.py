#import Agent
import gym
import matplotlib.pyplot as plt
import torch
from gym.wrappers import FrameStack
from environment.environment import EnvWrapper

if __name__ == '__main__':
    # Hyperparameters
    step_size = 4

    # Environment initialization
    env = EnvWrapper('procgen:procgen-starpilot-v0', step_size)

    obs, reward = env.step(env.env.action_space.sample())
    print(obs.shape, reward)
    while(True):


