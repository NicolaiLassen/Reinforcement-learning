from torch import nn


# Intrinsic Curiosity Model Reward
class ICM(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 1000) -> None:
        super(ICM, self).__init__()
