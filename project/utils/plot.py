import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

sns.set_theme(style="darkgrid")

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def smooth(y, box_pts):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


if __name__ == '__main__':
    rewards = torch.load('../ckpt_ppo_vit/starpilot_easy/rewards.ckpt')
    x = np.array([i * 2000 * 4 for i in range(len(rewards))])
    y = rewards.numpy()

    # Don't smooth ends
    sy = smooth(y, 12)
    sy[:4] = rewards[:4]
    sy[len(sy) - 4:len(sy)] = rewards[len(sy) - 4:len(sy)]

    d_smooth = {'Steps': x, 'Rewards': sy}
    df_smooth = pd.DataFrame(data=d_smooth)

    d = {'Steps': x, 'Rewards': y}
    df = pd.DataFrame(data=d)

    # #FF0000
    # #ffcc66
    sns.lineplot(x="Steps", y="Rewards", data=df, color='#FF0000', alpha=0.3, linewidth='2.6')
    sns.lineplot(x="Steps", y="Rewards", data=df_smooth, color='#FF0000', linewidth='1.5')
    plt.title("PPO VIT traning intrinsic rewards")
    # plt.yscale('log', base=2)
    plt.show()
