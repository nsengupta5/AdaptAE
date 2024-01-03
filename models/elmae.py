"""
File: elmae.py
Author: Nikhil Sengupta
Created on: November 6, 2023
Last Modified: January 5, 2024
Email: ns214@st-andrews.ac.uk

Description: 
    This file contains the implementation of the ELMAE model

License:
    This code is released under the MIT License
"""

import torch
import torch.nn as nn
from torch.linalg import lstsq
from torch.nn import functional as F
from util.util import assert_cond

class ELMAE(nn.Module):

    def __init__(self, activation_func, n_input_nodes, n_hidden_nodes, device):
        super().__init__()

        self.__name = "ELMAE"
        self.__n_input_nodes = n_input_nodes
        self.__n_hidden_nodes = n_hidden_nodes
        self.__penalty = 0.0001
        self.__device = device

        if activation_func == "tanh":
            self.__activation_func = torch.tanh
        else:
            raise ValueError("Activation function not supported")

        self.__alpha = nn.Parameter(torch.randn(n_input_nodes, n_hidden_nodes))
        nn.init.orthogonal_(self.__alpha)

        bias = torch.randn(n_hidden_nodes).to(device)
        bias = F.normalize(bias, p=2, dim=0)  # Normalize the bias vector to have unit norm
        self.__bias = nn.Parameter(bias)

        self.__beta = torch.zeros(n_hidden_nodes, n_input_nodes).to(device)

    """
    Predict the output of the network based on the input data
    :param test_data: The test data
    :type test_data: torch.Tensor
    :return: The predicted output
    :rtype: torch.Tensor
    """
    def predict(self, test_data):
        H = self.__activation_func(torch.matmul(test_data, self.__alpha) + self.__bias)
        return torch.matmul(H, self.__beta)

    """
    Predict the output of ELM-AE for sparse and compressed representations based on the input data
    :param train_data: The train data
    :type train_data: torch.Tensor
    """
    def calc_beta(self, train_data):
        assert_cond(train_data.shape[1] == self.__n_input_nodes, "Train data shape does not match the input nodes")

        H = self.__activation_func(torch.matmul(train_data, self.__alpha) + self.__bias)

        assert_cond(H.shape[1] == self.__n_hidden_nodes, "Hidden layer shape does not match the hidden nodes")
        assert_cond(H.shape[0] == train_data.shape[0], "Hidden layer shape does not match the train data")

        ident = torch.eye(self.__n_hidden_nodes).to(self.__device)
        H_TH = torch.matmul(H.T, H) 
        H_THI = H_TH + ident / self.__penalty
        H_THI_H_T = lstsq(H_THI, H.T).solution
        
        self.__beta = torch.matmul(H_THI_H_T, train_data)

    """
    Return the encoded representation of the input
    :param x: The input data
    :type x: torch.Tensor
    :return: The encoded representation of the input
    :rtype: torch.Tensor
    """
    def encoded_representation(self, x):
        return self.__activation_func(torch.matmul(x, self.__alpha) + self.__bias)

    """
    Return the input shape of the network
    :return: The input shape
    :rtype: tuple
    """
    @property
    def input_shape(self):
        return (self.__n_input_nodes,)

    """
    Return the hidden shape of the network
    :return: The hidden shape
    :rtype: tuple
    """
    @property
    def hidden_shape(self):
        return (self.__n_hidden_nodes,)

    """
    Return the device on which the network is running
    :return: The device
    :rtype: torch.device
    """
    @property
    def device(self):
        return self.__device

    """
    Return the name of the network
    :return: The name
    :rtype: str
    """
    @property
    def name(self):
        return self.__name
