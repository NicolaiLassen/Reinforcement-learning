from torch.distributions import Categorical

from project.agents.BaseAgent import BaseAgent
from project.environment.EnvWrapper import EnvWrapper
from project.models.policy_models import PolicyModelEncoder, PolicyModel


class PPOAgent(BaseAgent):
    def __init__(self, env: EnvWrapper, actor, critic, optimizer=None):
        self.env = env
        self.actor = actor
        self.critic = critic
        self.optimizer = optimizer

    def act(self, state):
        pass

    def eval(self):
        pass

    def train(self, num_episodes=5, num_steps=100):
        for i in range(num_episodes):

            s1, mask = self.env.reset()

            for j in range(num_steps):
                s = s1
                ## (S, N, E)
                ## TEMP BATCH SIZE
                s_enc = s.unsqueeze(0).permute(1, 0, 2)
                print(s_enc.shape)
                print(mask)
                act_probs = self.actor(s_enc, mask)
                act_dist = Categorical(act_probs)
                act = act_dist.sample()
                frame_seq, buffer, done = self.env.step(act)
                if done:
                    s1 = self.env.reset()
                    continue
                r_net = self.critic(s)


if __name__ == "__main__":
    seq_len = 100
    bach_size = 1
    env_wrapper = EnvWrapper('procgen:procgen-starpilot-v0', seq_len)

    actor = PolicyModelEncoder(seq_len, 64, 64, env_wrapper.env.action_space.n)
    critic = PolicyModel(seq_len, 64, 64, 1)

    agent = PPOAgent(env_wrapper, actor, critic)
    agent.train()
