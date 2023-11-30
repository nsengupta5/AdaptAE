import torch
import torch.nn as nn
from torch.linalg import lstsq
from torch.nn.functional import normalize
import logging

class ELMAE(nn.Module):

    def __init__(self, activation_func, loss_func, n_input_nodes, n_hidden_nodes, device):
        super().__init__()

        self.__n_input_nodes = n_input_nodes
        self.__n_hidden_nodes = n_hidden_nodes
        self.__penalty = 0.0001
        self.__device = device

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
        self.__beta = torch.zeros(n_hidden_nodes, n_input_nodes).to(device)

    """
    Predict the output of the network based on the input data
    :param test_data: The test data
    """
    def predict(self, test_data):
        H = self.__activation_func(torch.matmul(test_data, self.__alpha) + self.__bias)
        return torch.matmul(H, self.__beta)

    """
    Evaluate the network based on the test data and the predicted data
    :param test_data: The test data
    :param pred_data: The predicted data
    """
    def evaluate(self, test_data, pred_data):
        # Assert that:
        # 1. The test data and predicted data have the same shape
        # 2. The test data shape matches the input nodes
        # 3. The predicted data shape matches the input nodes
        assert_cond(test_data.shape[0] == pred_data.shape[0], "Test data and predicted data do not have the same shape")
        assert_cond(test_data.shape[1] == self.__n_input_nodes, "Test data shape does not match the input nodes")
        assert_cond(pred_data.shape[1] == self.__n_input_nodes, "Predicted data shape does not match the input nodes")
        loss = self.__loss_func(test_data, pred_data)
        accuracy = torch.sum(torch.argmax(self.predict(test_data), dim=1) == torch.argmax(pred_data, dim=1)) / len(pred_data) * 100
        return loss, accuracy

    """
    Predict the output of ELM-AE for sparse and compressed representations based on the input data
    :param train_data: The train data
    """
    def calc_beta_sparse(self, train_data):
        # Assert that the input data shape matches the input nodes
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
    Predict the output of ELM-AE for equal representations based on the input data
    :param train_data: The train data
    """
    def calc_beta_equal(self, train_data):
        # Assert that the input data shape matches the input nodes
        assert_cond(train_data.shape[1] == self.__n_input_nodes, "Train data shape does not match the input nodes")
        H = self.__activation_func(torch.matmul(train_data, self.__alpha) + self.__bias)
        assert_cond(H.shape[1] == self.__n_hidden_nodes, "Hidden layer shape does not match the hidden nodes")
        assert_cond(H.shape[0] == train_data.shape[0], "Hidden layer shape does not match the train data")

        self.__beta = lstsq(H, train_data).solution
        b_Tb = torch.round(torch.round(torch.matmul(self.__beta, self.__beta)))
        assert_cond(torch.allclose(b_Tb, torch.eye(self.__n_input_nodes).to(self.__device)), "Output layer parameters are not orthogonal")

    """
    Orthogonalize the hidden layer parameters
    """
    def orthogonalize_hidden_params(self):
        q, _ = torch.linalg.qr(self.__alpha)
        self.__alpha.data = q
        alpha_T = torch.transpose(self.__alpha, 0, 1)
        ident = torch.eye(self.__n_hidden_nodes).to(self.__device)
        a_Ta = torch.round(torch.matmul(alpha_T, self.__alpha))
        assert_cond(torch.allclose(a_Ta, ident), "Hidden layer parameters are not orthogonal")
        self.__bias.data = normalize(self.__bias, dim=0)
        b_Tb = torch.round(torch.matmul(self.__bias, self.__bias))
        assert_cond(b_Tb == 1, "Hidden layer bias is not normalized")

    """
    Return the input shape of the network
    """
    @property
    def input_shape(self):
        return (self.__n_input_nodes,)

    """
    Return the hidden shape of the network
    """
    @property
    def hidden_shape(self):
        return (self.__n_hidden_nodes,)

"""
Assert a condition and log the error if the condition is not met
"""
def assert_cond(condition, msg):
    try:
        assert condition, msg
    except AssertionError as e:
        logging.error(e)
        raise e
