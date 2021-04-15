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


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_mask):
        super(ActorCritic, self).__init__()

        # TODO
        self.encoder_out_dim = 3
        self.transformer_out_dim = 200

        # TODO TWEAK LAYERS?
        encoder_layers = [
            (1, 3, 6, 2, nn.BatchNorm2d(num_features=3)),
            (3, 3, 6, 2, nn.BatchNorm2d(num_features=3)),
            (3, self.encoder_out_dim, 4, 2, nn.BatchNorm2d(num_features=3)),
        ]


        # Reduce the input features for the transformer
        self.encoder = Encoder(layers=encoder_layers)
        # Use transformer to feature extract
        self.transformer = nn.Transformer(self.encoder_out_dim, dim_feedforward=self.transformer_out_dim)
        # Probs for action out
        self.fc_out = nn.Linear(self.transformer_out_dim, action_dim)

        ## We could split the model here? or in PPO?
        self.actor = nn.ModuleList()
        self.critic = nn.ModuleList()

    def forward(self, x):
        out = self.encoder(x)
        out = self.transformer(out)
        out = self.activation(out)
        return self.fc_out(out)
