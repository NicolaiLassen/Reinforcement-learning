import os

import gym
from torchvision import transforms

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


## TODO TAKE BATCH OF FRAMES 4-8
class EnvWrapper(gym.Env):
    def __init__(self, environment, seq_len=4, width: int = 64, height: int = 64, frameskip: int = 16):
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
        obs = None
        acc_reward = 0
        acc_done = False,
        # HACK TO FOR NOW TODO
        for i in range(4):
            obs, reward, done, info = self.env.step(action)
            acc_reward += reward
            if done and (not acc_done):  # fast hack if it overlaps with done run # TODO
                acc_done = done

        # PLOT HOW THIS IS GOING
        # plt.imshow(obs)
        # plt.show()
        frame = self.transformer(obs).view(-1, self.width * self.height)
        return frame, reward, done, info

    def reset(self):
        obs = self.env.reset()
        frame = self.transformer(obs).view(-1, self.width * self.height)
        return frame

    def render(self, **kwargs):
        return self.env.render()
