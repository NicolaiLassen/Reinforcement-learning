import argparse
import os

import torch

from agents.ppo_agent import PPOAgent
from environment.env_wrapper import EnvWrapper
from models.curiosity import IntrinsicCuriosityModule
from models.policy_models import PolicyModelConv, PolicyModel, PolicyModelVIT


def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", default="conv", help="PPO model")
    args = parser.parse_args()

    print(args.model)
    if args.model not in ["vit", "conv"]:
        exit(1)

    width = 64
    height = 64

    # Use static lr for testing purpose
    lr_actor = 0.0005
    lr_icm = 0.005
    lr_critic = 0.005

    # Hardcode for starpilot
    env_wrapper = EnvWrapper('procgen:procgen-starpilot-v0')
    create_dir("./ckpt_ppo_{}".format(args.model))
    create_dir("./ckpt_ppo_{}/starpilot_easy".format(args.model))

    actor = None
    if args.model == "vit":
        actor = PolicyModelVIT(width, height, env_wrapper.env.action_space.n).cuda()
    else:
        actor = PolicyModelConv(width, height, env_wrapper.env.action_space.n).cuda()

    critic = PolicyModel(width, height).cuda()
    icm = IntrinsicCuriosityModule(env_wrapper.env.action_space.n).cuda()

    optimizer = torch.optim.Adam([
        {'params': actor.parameters(), 'lr': lr_actor},
        {'params': icm.parameters(), 'lr': lr_icm},
        {'params': critic.parameters(), 'lr': lr_critic}
    ])

    # https://www.aicrowd.com/challenges/neurips-2020-procgen-competition
    # Challenge generalize for 8 million time steps cover 200 levels
    # max batch size GPU limit 64x64 * 2000 * nets_size
    agent = PPOAgent(env_wrapper, actor, critic, icm, optimizer, name=args.model)
    # SAVE MODEL EVERY 8000000 / 2000 / 100
    agent.train(2000, 8000000)
