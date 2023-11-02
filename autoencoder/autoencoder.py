import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, n_input_nodes, n_hidden_nodes):
        super().__init__()

        # Encoder with single hidden layer
        self.encoder = nn.Sequential(
            nn.Linear(n_input_nodes, n_hidden_nodes),
            nn.Sigmoid()
        )

        # Decoder with single hidden layer
        self.decoder = nn.Sequential(
            nn.Linear(n_hidden_nodes, n_input_nodes),
            nn.Sigmoid()
        )

    """
    Forward pass of the autoencoder
    :param x: The input data
    """
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
