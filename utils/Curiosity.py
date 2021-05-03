from torch import nn


# Intrinsic Curiosity Model Reward
class ICM(nn.Module):
    def __init__(self) -> None:
        super(ICM, self).__init__()
