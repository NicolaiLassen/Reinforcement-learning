import time
from statistics import mean
from typing import List

import torch

from agents.ppo_agent import PPOAgent
from environment.env_wrapper import EnvWrapper
from models.policy_models import PolicyModelVIT, PolicyModelConv
import matplotlib.pyplot as plt

def test_agent_ppo(actor, test_range: List[int] = [200, 400]):
    # TODO Test for each env on a test set: 200..400

    all_lvl_rewards = []
    all_lvl_steps_n = []
    for i in range(test_range[0], test_range[1]):
        env = EnvWrapper('procgen:procgen-starpilot-v0', start_level=i, num_levels=1)
        agent = PPOAgent(env, actor)

        lvl_rewards = []
        lvl_steps_n = []
        s1 = env.reset()
        for i in range(100):
            rewards = []
            steps_before_done = 0
            while True:
                s = s1
                steps_before_done += 1
                action, _, _ = agent.act(s)
                s1, r, d, _ = env.step(action)
                rewards.append(r)
                # plt.imshow(s1[0].cpu())
                # plt.show()
                # time.sleep(0.2)
                if d:
                    break
            lvl_rewards.append(sum(rewards))
            lvl_steps_n.append(steps_before_done)

        all_lvl_rewards.append(mean(lvl_rewards))
        all_lvl_rewards.append(mean(lvl_steps_n))

    return all_lvl_rewards, all_lvl_steps_n


if __name__ == '__main__':
    width = 64
    height = 64

    actor_conv = PolicyModelConv(width, height, 15).cuda()
    actor_conv.load_state_dict(torch.load("ckpt_ppo_conv/starpilot_easy/actor_950.ckpt"))
    actor_conv.eval()
    print(test_agent_ppo(actor_conv, test_range=[0, 200]))

    actor_vit = PolicyModelVIT(width, height, 15).cuda()
    actor_vit.load_state_dict(torch.load("ckpt_ppo_vit/starpilot_easy/actor_950.ckpt"))
    actor_vit.eval()
    print(test_agent_ppo(actor_conv, test_range=[0, 200]))
