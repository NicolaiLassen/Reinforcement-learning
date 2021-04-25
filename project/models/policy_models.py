import torch.nn as nn


class PolicyModel(nn.Module):
    def __init__(self, env_seq: int, width: int, height: int, action_dim: int = 1):
        super(PolicyModel, self).__init__()

        self.width = width
        self.height = height

        self.fc_1 = nn.Linear(width * height, width * height)
        self.fc_2 = nn.Linear(width * height, height)
        self.fc_out = nn.Linear(height, action_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        out = self.fc_1(x)
        out = self.activation(out)
        out = self.fc_1(out)
        out = self.activation(out)
        out = self.fc_2(out)
        out = self.activation(out)
        return self.fc_out(out)


class PolicyModelEncoder(nn.Module):
    def __init__(self, env_seq: int, width: int, height: int, action_dim: int):
        super(PolicyModelEncoder, self).__init__()

        self.width = width
        self.height = height

        # TODO SIZE of network
        # scale hypers
        # Should scale down dims
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=width * 8, nhead=2)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=4)

        # SHOULD US CON
        self.fc_in = nn.Linear(width * height, width * 8)
        self.fc_1 = nn.Linear(width * 8, width)
        self.fc_out = nn.Linear(height, action_dim)
        self.activation = nn.ReLU()
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x, mask=None):
        out = self.fc_in(x)
        out = self.encoder(out, mask)
        out = self.fc_1(out)
        out = self.activation(out)
        out = self.fc_out(out)
        return self.log_softmax(out)
