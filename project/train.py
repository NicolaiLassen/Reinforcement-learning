from project.environment.environment import EnvWrapper
from project.models.ActorCritic import ActorCritic, Encoder
from matplotlib import pyplot as plt
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

if __name__ == '__main__':
    # Hyperparameters
    step_size = 4

    # Environment initialization
    env = EnvWrapper('procgen:procgen-starpilot-v0', step_size)
    obs = env.reset()
    obs, reward, done, _ = env.step(env.env.action_space.sample())

    plt.imshow(obs[0])
    plt.show()

    actCritic = ActorCritic(state_dim=1, action_dim=env.env.action_space.n, action_mask="?")
    res = actCritic(obs)
    print(res)

#if __name__ == '__main__':
#
#    learning_rate = 0.005
#    episodes = 1000
#
#    model = ActorCritic()
#    optimizer = torch.optim.Adam([
#        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
#        {'params': self.policy.critic.parameters(), 'lr': lr_critic}
#    ])
#
#    agent = PPO(model, 1, optimizer)
#
#    # create traning loop
#    # env.render(mode="rgb_array")
#
#    obs = env.reset()
#    for i in range(episodes):
#        # use screen to generalize problems
#        screen = env.render(mode="rgb_array")
#        agent.act(screen)

