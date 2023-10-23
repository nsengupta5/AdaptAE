import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, n_input_nodes, n_hidden_nodes):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_input_nodes, n_hidden_nodes),
            nn.Sigmoid()
        )

        self.decoder = nn.Sequential(
            nn.Linear(n_hidden_nodes, n_input_nodes),
            nn.Sigmoid() # Need to know the ranges of the images, not always sigmoid
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

