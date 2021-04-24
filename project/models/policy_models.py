import torch.nn as nn


class PolicyModel(nn.Module):
    def __init__(self, env_seq: int, width: int, height: int, action_dim: int):
        super(PolicyModel, self).__init__()

        self.width = width
        self.height = height

        self.fc_1 = nn.Linear(width, height)
        self.fc_2 = nn.Linear(width * height, height)
        self.fc_out = nn.Linear(env_seq * height, action_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        out = self.fc_1(x)
        out = self.activation(out)
        out = out.view(-1, self.width * self.height)
        out = self.fc_2(out)
        out = self.activation(out)
        out = out.view(-1)
        out = self.fc_out(out)
        return out


class PolicyModelEncoder(nn.Module):
    def __init__(self, env_seq: int, width: int, height: int, action_dim: int):
        super(PolicyModelEncoder, self).__init__()

        self.width = width
        self.height = height

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=2)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=3)

        self.fc_1 = nn.Linear(width * height, height)
        self.fc_out = nn.Linear(env_seq * height, action_dim)
        self.activation = nn.ReLU()
        self.log_softmax = nn.LogSoftmax(dim=0)

    def forward(self, x, mask=None):
        out = self.encoder(x, mask)
        out = out.view(-1, self.width * self.height)
        out = self.fc_1(out)
        out = self.activation(out)
        out = out.view(-1)
        out = self.fc_out(out)
        return self.log_softmax(out)
