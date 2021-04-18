import torch.nn as nn


## FlAG ## V0.0.1 Reducer Transformer
class Encoder(nn.Module):
    def __init__(self, layers):
        super(Encoder, self).__init__()

        def encoder_conv_block(n_in, n_out, kernel_size, stride, activation):
            return nn.Sequential(
                nn.Conv2d(n_in, n_out, kernel_size=kernel_size, stride=stride),
                activation
            )

        self.conv_blocks = nn.ModuleList()

        for n_in, n_out, kernel_size, stride, activation in layers:
            self.conv_blocks.append(encoder_conv_block(n_in, n_out, kernel_size, stride, activation))

        self.encoder = nn.Sequential(*self.conv_blocks)

    def forward(self, x):
        return self.encoder(x)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, actor):
        super(Actor, self).__init__()

        # TODO
        self.encoder_out_dim = 3
        self.actor = actor

        # TODO TWEAK LAYERS?
        encoder_layers = [
            (1, 3, 6, 2, nn.BatchNorm2d(num_features=3)),
            (3, 3, 6, 2, nn.BatchNorm2d(num_features=3)),
            (3, self.encoder_out_dim, 4, 2, nn.BatchNorm2d(num_features=3)),
        ]

        # Reduce the input features for the transformer
        # self.encoder = Encoder(layers=encoder_layers)
        # Use transformer to feature extract
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=2)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=3)
        # Probs for action out
        self.fc_1 = nn.Linear(64*64*3, 64)
        self.fc_out = nn.Linear(64, action_dim)
        self.activation = nn.ReLU()
        self.log_softmax = nn.LogSoftmax()

    def forward(self, x):
        out = x
        # out = self.encoder(x)
        out = self.encoder(out).view(-1)
        out = self.fc_1(out)
        out = self.activation(out)
        out = self.fc_out(out)
        if self.actor:
            return self.log_softmax(out)
        else:
            return out
