import os

import matplotlib.pyplot as plt
import torch

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
if __name__ == '__main__':
    rewards = torch.load('../ckpt_ppo/rewards.ckpt')
    print(rewards)
    print(len(rewards))
    print(len(rewards) * 2000)

    plt.plot(rewards.numpy()[10:], color='b')
    plt.show()
