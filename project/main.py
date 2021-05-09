import os

import torch

from agents.ppo_agent import PPOAgent
from environment.env_wrapper import EnvWrapper
from models.curiosity import IntrinsicCuriosityModule
from models.policy_models import PolicyModelEncoder, PolicyModel


def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def test_agent_ppo():
    # TODO Test for each env start_0..500
    for i in range(500):
        env = EnvWrapper('procgen:procgen-starpilot-v0', start_level=i, num_levels=1)
        width = 64
        height = 64
        actor = PolicyModelEncoder(width, height, env.env.action_space.n).cuda()
        agent = PPOAgent(env, actor)
        agent.load_actor("ckpt_ppo/actor_2000.ckpt")
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


def test_agent_dqn():
    # TODO Test for each env 0..500
    for i in range(500):
        env = EnvWrapper('procgen:procgen-starpilot-v0', start_level=i, num_levels=1)


if __name__ == "__main__":
    create_dir("./ckpt_ppo")
    bach_size = 1
    width = 64
    height = 64

    lr_actor = 0.0005
    lr_icm = 0.001
    lr_critic = 0.001

    env_wrapper = EnvWrapper('procgen:procgen-starpilot-v0')

    actor = PolicyModelEncoder(width, height, env_wrapper.env.action_space.n).cuda()
    critic = PolicyModel(width, height).cuda()
    icm = IntrinsicCuriosityModule(env_wrapper.env.action_space.n).cuda()

    optimizer = torch.optim.Adam([
        {'params': actor.parameters(), 'lr': lr_actor},
        {'params': icm.parameters(), 'lr': lr_icm},
        {'params': critic.parameters(), 'lr': lr_critic}
    ])

    agent = PPOAgent(env_wrapper, actor, critic, icm, optimizer)
    agent.train(2000, 200000000)
