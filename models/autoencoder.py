"""
File: autoencoder.py
Author: Nikhil Sengupta
Created on: November 6, 2023
Last Modified: January 5, 2024
Email: ns214@st-andrews.ac.uk

Description: 
    This file contains the implementation of a vanilla autoencoder.

License:
    This code is released under the MIT License
"""

import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, n_input_nodes, n_hidden_nodes, device):
        super().__init__()

        self.__name = "Autoencoder"
        self.__device = device

        self.__n_input_nodes = n_input_nodes
        self.__n_hidden_nodes = n_hidden_nodes

        # Encoder with single hidden layer
        self.encoder = nn.Sequential(
            nn.Linear(n_input_nodes, n_hidden_nodes),
        )

        # Decoder with single hidden layer
        self.decoder = nn.Sequential(
            nn.Linear(n_hidden_nodes, n_input_nodes),
            nn.Tanh()
        )

    """
    Forward pass of the autoencoder
    :param x: The input data
    :type x: torch.Tensor
    :return: The output of the autoencoder
    :rtype: torch.Tensor
    """
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    """
    Return the encoded representation of the input
    :param x: The input data
    :type x: torch.Tensor
    :return: The encoded representation of the input
    :rtype: torch.Tensor
    """
    def encoded_representation(self, x):
        return self.encoder(x)

    """
    Return the input shape of the autoencoder
    :return: The input shape of the autoencoder
    :rtype: tuple
    """
    @property
    def input_shape(self):
        return (self.__n_input_nodes,)

    """
    Return the hidden shape of the autoencoder
    :return: The hidden shape of the autoencoder
    :rtype: tuple
    """
    @property
    def hidden_shape(self):
        return (self.__n_hidden_nodes,)

    """
    Return the device on which the autoencoder is running
    :return: The device on which the autoencoder is running
    :rtype: torch.device
    """
    @property
    def device(self):
        return self.__device

    """
    Return the name of the autoencoder
    :return: The name of the autoencoder
    :rtype: str
    """
    @property
    def name(self):
        return self.__name
