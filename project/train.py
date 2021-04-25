import torch

from project.agents.PPOAgent import PPOAgent
from project.environment.EnvWrapper import EnvWrapper
from project.models.policy_models import PolicyModelEncoder, PolicyModel

if __name__ == "__main__":
    seq_len = 1
    bach_size = 1
    width = 64
    height = 64

    lr_actor = 0.0005
    lr_critic = 0.001

    # SWITCH THIS IN EQ BATCHES - NO cheating and getting good at only one thing
    env_wrapper = EnvWrapper('procgen:procgen-starpilot-v0', seq_len)

    actor = PolicyModelEncoder(seq_len, width, height, env_wrapper.env.action_space.n)
    critic = PolicyModel(seq_len, width, height)

    optimizer = torch.optim.Adam([
        {'params': actor.parameters(), 'lr': lr_actor},
        {'params': critic.parameters(), 'lr': lr_critic}
    ])

    print(actor)
    print(critic)
    agent = PPOAgent(env_wrapper, actor, critic, optimizer)
    # TODO: More and smaller batches
    agent.train(400, 100000)
