if __name__ == "__main__":
    # env = EnvWrapper('procgen:procgen-starpilot-v0', 1)
    env = gym.make('procgen:procgen-starpilot-v0')

    actor = PolicyModel(64, 64, env.action_space.n, True)
    critic = PolicyModel(64, 64, 1, False)

    actorCritic = PPOAgent(env, actor, critic)
    actorCritic.train()