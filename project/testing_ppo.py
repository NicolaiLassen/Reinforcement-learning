from typing import List

import torch

from agents.ppo_agent import PPOAgent
from environment.env_wrapper import EnvWrapper
from models.policy_models import PolicyModelVIT, PolicyModelConv


def test_agent_ppo(actor, test_range: List[int] = [200, 400]):
    # TODO Test for each env on a test set: 200..400

    all_rewards = []
    all_steps_n = []
    for i in range(test_range[0], test_range[1]):
        env = EnvWrapper('procgen:procgen-starpilot-v0', start_level=i, num_levels=1)
        agent = PPOAgent(env, actor)
        s1 = env.reset()
        rewards = []
        steps_before_done = 0
        while True:
            s = s1
            steps_before_done += 1
            action, _, _ = agent.act(s)
            s1, r, d, _ = env.step(action)
            rewards.append(r)
            if d:
                break
        all_rewards.append(sum(rewards))
        all_steps_n.append(steps_before_done)

    return all_rewards


if __name__ == '__main__':
    width = 64
    height = 64

    actor_conv = PolicyModelConv(width, height, 15).cuda()
    actor_conv.load_state_dict(torch.load("ckpt_ppo_conv/starpilot_easy/actor_950.ckpt"))
    actor_conv.eval()
    print(sum(test_agent_ppo(actor_conv, test_range=[200, 400])))

    # actor_vit = PolicyModelVIT(width, height, 15).eval()

    # actor = PolicyModelConv(width, height, env_wrapper.env.action_space.n).cuda()
