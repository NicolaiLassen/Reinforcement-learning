from typing import List

import torch

from agents.ppo_agent import PPOAgent
from environment.env_wrapper import EnvWrapper
from models.policy_models import PolicyModelVIT


def test_agent_ppo(actor, env, test_range: List[int] = [200, 400]):
    # TODO Test for each env on a test set: 200..400
    for i in range(test_range[0], test_range[1]):
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
        print("Level: {}, N Steps: {}, Total Reward: {}".format(i, steps_before_done, sum(rewards)))


if __name__ == '__main__':
    width = 64
    height = 64

    env = EnvWrapper('procgen:procgen-starpilot-v0', start_level=i, num_levels=1)
    actor = PolicyModelVIT(width, height, env.env.action_space.n).cuda()
    actor.eval()
    actor.load_state_dict(torch.load("path"))
    test_agent_ppo(actor, env)

    # actor = PolicyModelConv(width, height, env_wrapper.env.action_space.n).cuda()
