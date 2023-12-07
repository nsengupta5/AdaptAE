"""
File: autoencoder.py
Author: Nikhil Sengupta
Created on: November 6, 2023
Last Modified: December 12, 2023
Email: ns214@st-andrews.ac.uk

Description: 
    This file contains my implementation of an autoencoder with a single hidden 
    layer.

License:
    This code is released under the MIT License
"""

import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, n_input_nodes, n_hidden_nodes):
        super().__init__()

        self.__n_input_nodes = n_input_nodes
        self.__n_hidden_nodes = n_hidden_nodes

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

    """
    Return the input shape of the autoencoder
    :return: The input shape of the autoencoder
    """
    @property
    def input_shape(self):
        return (self.__n_input_nodes,)

    """
    Return the hidden shape of the autoencoder
    :return: The hidden shape of the autoencoder
    """
    @property
    def hidden_shape(self):
        return (self.__n_hidden_nodes,)
