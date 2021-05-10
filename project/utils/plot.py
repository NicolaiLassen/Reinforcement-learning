import os

import matplotlib.pyplot as plt
import numpy as np
import torch

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def smooth(y, box_pts):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


if __name__ == '__main__':
    rewards = torch.load('../ckpt_ppo_vit/starpilot_easy/rewards.ckpt')
    # rewards = torch.load('../ckpt_ppo_vit/starpilot_easy/intrinsic_rewards.ckpt')[15:]
    # rewards = torch.load('../ckpt_ppo_vit/starpilot_easy/rewards.ckpt')

    print(len(rewards))
    x = np.array([i * 5000 for i in range(len(rewards))])
    y = rewards.numpy()

    # Don't smooth ends
    sy = smooth(y, 12)
    sy[:4] = rewards[:4]
    sy[len(sy) - 4:len(sy)] = rewards[len(sy) - 4:len(sy)]

    # Intrinsic
    plt.title("PPO VIT Rewards")
    plt.plot(x, y, color='r', alpha=0.25, linewidth='2.5')
    plt.plot(x, sy, color='r', linewidth='1.5')
    plt.xlabel("Steps")
    plt.ylabel("Rewards")
    plt.show()
#    plt.plot(value_np_smooth, color='r')
#    plt.show()
