from project.agents.PPOAgent import PPOAgent
from project.environment.EnvWrapper import EnvWrapper
from project.models.policy_models import PolicyModel

if __name__ == "__main__":
    env = EnvWrapper('procgen:procgen-starpilot-v0', 4)
    # env = gym.make('procgen:procgen-starpilot-v0')

    # actor = PolicyModel(64, 64, env.env.action_space.n, True)
    # critic = PolicyModel(64, 64, 1, False)

    seq_frames = env.step(env.env.action_space.sample())
    print(seq_frames.size)

    # actorCritic = PPOAgent(env, actor, critic)
    # actorCritic.train()