import torch

from project.agents.ppo_agent import PPOAgent
from project.environment.env_wrapper import EnvWrapper
from project.models.policy_models import PolicyModelEncoder, PolicyModel
from utils.curiosity import IntrinsicCuriosityModule

if __name__ == "__main__":
    bach_size = 1
    width = 64
    height = 64

    lr_actor = 0.0005
    lr_icm = 0.001
    lr_critic = 0.001

    # SWITCH THIS IN EQ BATCHES - NO cheating and getting good at only one thing
    env_wrapper = EnvWrapper('procgen:procgen-starpilot-v0')

    actor = PolicyModelEncoder(width, height, env_wrapper.env.action_space.n).cuda()
    critic = PolicyModel(width, height).cuda()
    icm = IntrinsicCuriosityModule(env_wrapper.env.action_space.n).cuda()

    optimizer_actor = torch.optim.Adam([
        {'params': actor.parameters(), 'lr': lr_actor},
        {'params': icm.parameters(), 'lr': lr_icm}
    ])
    optimizer_critic = torch.optim.Adam(critic.parameters(), lr=lr_critic)

    agent = PPOAgent(env_wrapper, actor, critic, icm, optimizer_actor, optimizer_critic)
    # 20000
    agent.train(400, 200000000)
