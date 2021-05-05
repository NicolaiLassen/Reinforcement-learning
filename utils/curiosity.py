import torch
import torch.nn.functional as F
from torch import nn


# Intrinsic Curiosity Model Reward
class ICM(nn.Module):
    def __init__(self, action_space_n: int, width: int = 50, height: int = 50) -> None:
        super(ICM, self).__init__()

        self.head = nn.Sequential()

        self.state_size = width * height

        self.forward_model = nn.Sequential(
            nn.Linear(self.state_size + action_space_n, 512),
            nn.ReLU(),
            nn.Linear(512, self.feature_size)
        )

        self.inverse_model = nn.Sequential(
            nn.Linear(self.state_size * 2, 512),
            nn.ReLU(),
            nn.Linear(512, action_space_n)
        )

    def forward(self, state, next_state, action):
        phi_t = self.head(state)
        phi_t1 = self.head(next_state)
        phi_t1_hat = self.forward_model(torch.cat((phi_t, action), 1))
        a_t_hat = F.softmax(self.inverse_model(torch.cat((phi_t, phi_t1), 1)), dim=-1)
        return a_t_hat, phi_t1_hat, phi_t1, phi_t
