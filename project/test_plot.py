import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch

sns.set()

sns.set_theme(style="darkgrid")
if __name__ == '__main__':
    conv_rewards_train = torch.load('ckpt_train/conv_test_rewards.ckpt').numpy()
    conv_rewards_test = torch.load('ckpt_test/conv_test_rewards.ckpt').numpy()
    print(conv_rewards_train.mean())

    vit_rewards_train = torch.load('ckpt_train/vit_test_rewards.ckpt').numpy()
    vit_rewards_test = torch.load('ckpt_test/vit_test_rewards.ckpt').numpy()
    print(vit_rewards_train.mean())

    levels = [i for i in range(0, 50)]
    levels = levels + levels

    convs = ['conv' for i in range(0, 50)]
    vits = ['vit' for i in range(0, 50)]

    model_names = convs + vits
    models_mean = list(conv_rewards_train)[0:50] + list(vit_rewards_train)[0:50]

    # #FF0000
    # #ffcc66

    # bar chart 1 -> top bars (group of 'smoker=No')
    d_train = {'Level': levels, 'Mean reward': models_mean, 'model': model_names}
    df_train = pd.DataFrame(data=d_train).set_index('Level')

    bar1 = sns.barplot(x="Level", y="Mean reward", data=d_train, hue='model', palette=['#FF0000', '#ffcc66'])

    top_bar = mpatches.Patch(color='#FF0000', label='Conv')
    bottom_bar = mpatches.Patch(color='#ffcc66', label='VIT')
    plt.legend(handles=[top_bar, bottom_bar])

    bar1.set(xticklabels=[])
    plt.title("Training maps 0...50")
    bar1.set(ylabel='Mean rewards')
    plt.show()
