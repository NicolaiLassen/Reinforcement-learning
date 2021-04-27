import torch.nn as nn

# Critic Model
from project.models.SqueezeNet import SqueezeNet


class PolicyModel(nn.Module):
    def __init__(self, env_seq: int, width: int, height: int, action_dim: int = 1, motion_blur: int = 4):
        super(PolicyModel, self).__init__()

        self.width = width
        self.height = height
        self.motion_blur = motion_blur

        self.fc_1 = nn.Linear(motion_blur * width * height, width * height)
        self.fc_2 = nn.Linear(width * height, width * height)
        self.fc_3 = nn.Linear(width * height, height)
        self.fc_out = nn.Linear(height, action_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        out = x.view(-1, self.motion_blur * self.width * self.height)
        out = self.fc_1(out)
        out = self.activation(out)
        out = self.fc_2(out)
        out = self.activation(out)
        out = self.fc_3(out)
        out = self.activation(out)
        return self.fc_out(out)


# Actor Model
class PolicyModelEncoder(nn.Module):
    def __init__(self, env_seq: int, width: int, height: int, action_dim: int, motion_blur: int = 4):
        super(PolicyModelEncoder, self).__init__()

        self.width = width
        self.height = height
        self.motion_blur = motion_blur

        self.encoder_squeeze = SqueezeNet(motion_blur, width * 8)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=width * 8, nhead=2)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=4)

        self.fc_out = nn.Linear(width * 8, action_dim)
        self.activation = nn.ReLU()
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x, mask=None):
        out = x.view(-1, self.motion_blur, self.width, self.height)
        out = self.encoder_squeeze(out)
        out = out.unsqueeze(0).permute(1, 0, 2)
        out = self.encoder(out, mask)
        out = self.fc_out(out)
        return self.log_softmax(out)
