import torch
from torch import nn
from torch.linalg import pinv
import logging

class OSELM(nn.Module):

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
        self.__u = nn.Parameter(torch.zeros(n_hidden_nodes, n_hidden_nodes))
        self.__v = nn.Parameter(torch.zeros(n_hidden_nodes, n_input_nodes))
        self.__device = device

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
    Initialize the network based on the input data
    :param data: The input data for initialization phase
    """
    def init_phase(self, data):
        # Assert that the input data shape matches the input nodes
        assert_cond(data.shape[1] == self.__n_input_nodes, "Input data shape does not match the input nodes")
        # Assert that the hidden layer shape matches the hidden nodes
        H = self.__activation_func(torch.matmul(data, self.__alpha) + self.__bias)
        assert_cond(H.shape[1] == self.__n_hidden_nodes, "Hidden layer shape does not match the hidden nodes")
        assert_cond(H.shape[0] == data.shape[0], "Hidden layer shape does not match number of samples")

        H_T = torch.transpose(H, 0, 1)
        self.__p = pinv(torch.matmul(H_T, H))
        pH_T = torch.matmul(self.__p, H_T)
        self.__beta = torch.matmul(pH_T, data)
        return self.__beta

    """
    Sequentially train the network based on the input data
    :param data: The input data for sequential training
    :param mode: The mode of training, either "batch" or "sample"
    """
    def seq_phase(self, data, mode):
        # Assert that the hidden layer shape matches the hidden nodes
        H = self.__activation_func(torch.matmul(data, self.__alpha) + self.__bias)

        if mode == "batch":
            H_T = torch.transpose(H, 0, 1)
            assert_cond(H.shape[1] == self.__n_hidden_nodes, "Hidden layer shape does not match the hidden nodes")
            assert_cond(data.shape[1] == self.__n_input_nodes, "Input data shape does not match the input nodes")
            batch_size = data.shape[0]
            self.calc_p_batch(batch_size, H, H_T)
            self.calc_beta_batch(data, H, H_T)
        elif mode == "sample":
            H = H.unsqueeze(1)
            H_T = torch.transpose(H, 0, 1)
            assert_cond(H_T.shape[1] == self.__n_hidden_nodes, "Hidden layer shape does not match the hidden nodes")
            self.calc_p_sample(H, H_T)
            self.calc_beta_sample(data, H, H_T)
        else:
            raise ValueError("Mode not supported")

        return self.__beta

    """
    Calculate the p of the network based on batch of input data
    :param batch_size: The size of the batch
    :param H: The hidden layer output matrix
    :param H_T: The transpose of the hidden layer output matrix
    """
    def calc_p_batch(self, batch_size, H, H_T):
        PH_T = torch.matmul(self.__p, H_T)
        I = torch.eye(batch_size).to(self.__device)
        HPH_T_Inv = pinv(torch.matmul(H, torch.matmul(self.__p, H_T)) + I)
        HP = torch.matmul(H, self.__p)
        self.__p -= torch.matmul(torch.matmul(PH_T, HPH_T_Inv), HP)

    """
    Calculate the beta of the network based on batch of input data
    :param batch: The batch of input data
    :param H: The hidden layer output matrix
    :param H_T: The transpose of the hidden layer output matrix
    """
    def calc_beta_batch(self, batch, H, H_T):
        T = torch.transpose(batch, 0, 1)
        THB = T - torch.matmul(H, self.__beta)
        self.__beta += torch.matmul(torch.matmul(self.__p, H_T), THB)

    """
    Calculate the p of the network based on sample of input data
    :param H: The hidden layer output matrix
    :param H_T: The transpose of the hidden layer output matrix
    """
    def calc_p_sample(self, H, H_T):
        with torch.no_grad():
            PH = torch.matmul(self.__p, H)
            PHH_T = torch.matmul(PH, H_T)
            del PH
            PHH_TP = torch.matmul(PHH_T, self.__p)
            del PHH_T
            H_TPH = torch.matmul(H_T, torch.matmul(self.__p, H))
            del H_T
            self.__p -= torch.div(PHH_TP, 1 + H_TPH)

    """
    Calculate the beta of the network based on sample of input data
    :param sample: The sample of input data
    :param H: The hidden layer output matrix
    """
    def calc_beta_sample(self, sample, H, H_T):
        with torch.no_grad():
            THB = sample - torch.matmul(H_T, self.__beta)
            PH_T = torch.matmul(self.__p, H)
            self.__beta += torch.matmul(PH_T, THB)

    """
    Calculate the intermediate u of the network as part of E^2LM
    """
    def calc_interm_u(self):
        self.__u = pinv(self.__p)

    """
    Calculate the intermediate v of the network as part of E^2LM
    """
    def calc_interm_v(self):
        self.__v = torch.matmul(self.__u, self.__beta)

    """
    Integrate the intermediate u and v from another network into this network as part of E^2LM
    """
    def integrate(self, u, v):
        self.__u += u
        self.__v += v
        self.__p = pinv(self.__u)
        self.__beta = torch.matmul(pinv(self.__u), self.__v)

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
