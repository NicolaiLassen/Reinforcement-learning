import torch.nn as nn
import torch.nn.functional as F

# Ref https://github.com/lukemelas/EfficientNet-PyTorch


# Critic Model
from vit_pytorch import ViT


class PolicyModel(nn.Module):
    def __init__(self, width: int, height: int, action_dim: int = 1, motion_blur: int = 4):
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
    def __init__(self, width: int, height: int, action_dim: int, motion_blur: int = 4):
        super(PolicyModelEncoder, self).__init__()

        self.width = width
        self.height = height
        self.motion_blur = motion_blur

        self.encoder_out_dim = 512
        self.image_encoder = ViT(
            image_size=64,
            patch_size=16,
            num_classes=self.encoder_out_dim,
            dim=512,
            depth=4,
            channels=4,
            heads=4,
            mlp_dim=1024,
            dropout=0.1,
            emb_dropout=0.1
        )
        self.fc_1 = nn.Linear(self.encoder_out_dim, self.encoder_out_dim)
        self.fc_out = nn.Linear(self.encoder_out_dim, action_dim)
        self.activation = nn.ReLU()

    def forward(self, x, mask=None):
        out = x.view(-1, self.motion_blur, self.width, self.height)
        out = self.image_encoder(out)
        out = self.fc_1(out)
        out = self.activation(out)
        out = self.fc_out(out)
        return F.log_softmax(out, dim=-1)
