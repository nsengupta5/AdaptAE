"""
File: pselmae.py
Author: Nikhil Sengupta
Created on: November 6, 2023
Last Modified: December 12, 2023
Email: ns214@st-andrews.ac.uk

Description: 
    This file contains 

License:
    This code is released under the MIT License
"""

import torch
from util.util import assert_cond
from torch import nn
from torch.linalg import pinv

class PSELMAE(nn.Module):

    def __init__(self, activation_func, loss_func, n_input_nodes, n_hidden_nodes, device):
        super().__init__()

        self.__n_input_nodes = n_input_nodes
        self.__n_hidden_nodes = n_hidden_nodes

        if activation_func == "sigmoid":
            self.__activation_func = torch.sigmoid
        else:
            raise ValueError("Activation function not supported")

        if loss_func == "mse":
            self.__loss_func = nn.MSELoss()
        elif loss_func == "cross_entropy":
            self.__loss_func = nn.CrossEntropyLoss()
        else:
            raise ValueError("Loss function not supported")

        self.__alpha = nn.Parameter(torch.randn(n_input_nodes, n_hidden_nodes))
        self.__bias = nn.Parameter(torch.randn(n_hidden_nodes))

        self.__p = torch.zeros(n_hidden_nodes, n_hidden_nodes).to(device)
        self.__beta = torch.zeros(n_hidden_nodes, n_input_nodes).to(device)
        self.__device = device

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
    Evaluate the network based on the test data and the predicted data
    :param test_data: The test data
    :type test_data: torch.Tensor
    :param pred_data: The predicted data
    :type pred_data: torch.Tensor
    :return: The loss and accuracy
    :rtype loss: torch.Tensor
    :rtype accuracy: torch.Tensor
    """
    def evaluate(self, test_data, pred_data):
        assert_cond(test_data.shape[0] == pred_data.shape[0], "Test data and predicted data do not have the same shape")
        assert_cond(test_data.shape[1] == self.__n_input_nodes, "Test data shape does not match the input nodes")
        assert_cond(pred_data.shape[1] == self.__n_input_nodes, "Predicted data shape does not match the input nodes")
        loss = self.__loss_func(test_data, pred_data)
        accuracy = torch.sum(torch.argmax(self.predict(test_data), dim=1) == torch.argmax(pred_data, dim=1)) / len(pred_data) * 100
        return loss, accuracy

    """
    Initialize the network based on the input data
    :param data: The input data for initialization phase
    :type data: torch.Tensor
    :return: The network after initialization phase
    :rtype: torch.Tensor
    """
    def init_phase(self, data):
        assert_cond(data.shape[1] == self.__n_input_nodes, "Input data shape does not match the input nodes")
        H = self.__activation_func(torch.matmul(data, self.__alpha) + self.__bias)
        assert_cond(H.shape[1] == self.__n_hidden_nodes, "Hidden layer shape does not match the hidden nodes")
        assert_cond(H.shape[0] == data.shape[0], "Hidden layer shape does not match number of samples")

        self.__p = pinv(torch.matmul(H.T, H))
        pH_T = torch.matmul(self.__p, H.T)
        del H
        self.__beta = torch.matmul(pH_T, data)
        del pH_T
        return self.__beta

    """
    Sequentially train the network based on the input data
    :param data: The input data for sequential training
    :type data: torch.Tensor
    :param mode: The mode of training, either "batch" or "sample"
    :type mode: str
    :return: The network after sequential training
    :rtype: torch.Tensor
    """
    def seq_phase(self, data, mode):
        # Assert that the hidden layer shape matches the hidden nodes
        H = self.__activation_func(torch.matmul(data, self.__alpha) + self.__bias)

        if mode == "batch":
            assert_cond(H.shape[1] == self.__n_hidden_nodes, "Hidden layer shape does not match the hidden nodes")
            assert_cond(data.shape[1] == self.__n_input_nodes, "Input data shape does not match the input nodes")
            batch_size = data.shape[0]
            self.calc_p_batch(batch_size, H)
            self.calc_beta_batch(data, H)
        elif mode == "sample":
            assert_cond(H.shape[1] == self.__n_hidden_nodes, "Hidden layer shape does not match the hidden nodes")
            self.calc_p_sample(H.T)
            self.calc_beta_sample(data, H.T)
        else:
            raise ValueError("Mode not supported")

        return self.__beta

    """
    Calculate the p of the network based on batch of input data
    :param batch_size: The size of the batch
    :type batch_size: int
    :param H: The hidden layer output matrix
    :type H: torch.Tensor
    """
    def calc_p_batch(self, batch_size, H):
        PH_T = torch.matmul(self.__p, H.T)
        I = torch.eye(batch_size).to(self.__device)
        HPH_T_Inv = pinv(torch.matmul(H, torch.matmul(self.__p, H.T)) + I)
        del I
        HP = torch.matmul(H, self.__p)
        self.__p -= torch.matmul(torch.matmul(PH_T, HPH_T_Inv), HP)

    """
    Calculate the beta of the network based on batch of input data
    :param batch: The batch of input data
    :type batch: torch.Tensor
    :param H: The hidden layer output matrix
    :type H: torch.Tensor
    """
    def calc_beta_batch(self, batch, H):
        THB = batch - torch.matmul(H, self.__beta)
        self.__beta += torch.matmul(torch.matmul(self.__p, H.T), THB)

    """
    Calculate the p of the network based on sample of input data
    :param H: The hidden layer output matrix
    :type H: torch.Tensor
    """
    def calc_p_sample(self, H):
        with torch.no_grad():
            PH = torch.matmul(self.__p, H)
            PHH_T = torch.matmul(PH, H.T)
            del PH
            PHH_TP = torch.matmul(PHH_T, self.__p)
            del PHH_T
            H_TPH = torch.matmul(H.T, torch.matmul(self.__p, H))
            self.__p -= torch.div(PHH_TP, 1 + H_TPH)

    """
    Calculate the beta of the network based on sample of input data
    :param sample: The sample of input data
    :type sample: torch.Tensor
    :param H: The hidden layer output matrix
    :type H: torch.Tensor
    """
    def calc_beta_sample(self, sample, H):
        with torch.no_grad():
            THB = sample - torch.matmul(H.T, self.__beta)
            PH_T = torch.matmul(self.__p, H)
            self.__beta += torch.matmul(PH_T, THB)

    """
    Return the input shape of the network
    :return: The input shape of the network
    :rtype: tuple
    """
    @property
    def input_shape(self):
        return (self.__n_input_nodes,)

    """
    Return the hidden shape of the network
    :return: The hidden shape of the network
    :rtype: tuple
    """
    @property
    def hidden_shape(self):
        return (self.__n_hidden_nodes,)
