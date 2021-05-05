import torch

from project.agents.ppo_agent import PPOAgent
from project.environment.env_wrapper import EnvWrapper
from project.models.policy_models import PolicyModelEncoder, PolicyModel

if __name__ == "__main__":
    seq_len = 1
    bach_size = 1
    width = 64
    height = 64

    lr_actor = 0.0005
    lr_critic = 0.001