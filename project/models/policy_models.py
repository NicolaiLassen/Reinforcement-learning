import torch.nn as nn
import torch.nn.functional as F
from vit_pytorch import ViT


# Critic Model
class PolicyModel(nn.Module):
    def __init__(self, width: int, height: int, action_dim: int = 1, motion_blur: int = 4):
        super(PolicyModel, self).__init__()

        self.width = width
        self.height = height
        self.motion_blur = motion_blur
        self.embed_dim = 128

        self.fc_1 = nn.Linear(width * height, self.embed_dim)
        self.fc_2 = nn.Linear(motion_blur * self.embed_dim, self.embed_dim)
        self.fc_out = nn.Linear(self.embed_dim, action_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        out = x.view(-1, self.motion_blur, self.width * self.height)
        out = self.fc_1(out)
        out = self.activation(out)
        out = out.view(-1, self.motion_blur * self.embed_dim)
        out = self.fc_2(out)
        out = self.activation(out)
        return self.fc_out(out)


# Actor Model
class PolicyModelVIT(nn.Module):
    def __init__(self, width: int, height: int, action_dim: int, motion_blur: int = 4):
        super(PolicyModelVIT, self).__init__()

        self.width = width
        self.height = height
        self.motion_blur = motion_blur

        self.image_encoder = ViT(
            image_size=64,
            patch_size=8,
            num_classes=action_dim,
            dim=128,
            depth=2,
            channels=4,
            heads=2,
            mlp_dim=128,
            dropout=0,
            emb_dropout=0
        )

    def forward(self, x):
        out = self.image_encoder(x)
        return F.log_softmax(out, dim=-1)


class PolicyModelConv(nn.Module):
    def __init__(self, width: int, height: int, action_dim: int, motion_blur: int = 4):
        super(PolicyModelConv, self).__init__()

        self.width = width
        self.height = height
        self.motion_blur = motion_blur
        self.embed_dim = 512

        # Natural Head
        self.conv1 = nn.Conv2d(motion_blur, 32, kernel_size=(8, 8), stride=(4, 4))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2))
        self.conv3 = nn.Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1))
        self.activation = nn.ReLU()

        self.fc_1 = nn.Linear(32 * 4 * self.motion_blur, self.embed_dim)
        self.fc_out = nn.Linear(self.embed_dim, action_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        out = self.activation(self.conv1(x))
        out = self.activation(self.conv2(out))
        out = self.activation(self.conv3(out))
        out = out.view(-1, self.motion_blur * 32 * 4)

        out = self.fc_1(out)
        out = self.activation(out)
        out = self.fc_out(out)
        return F.log_softmax(out, dim=-1)
