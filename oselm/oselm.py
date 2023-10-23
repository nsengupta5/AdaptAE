import torch
from torch import nn
from torch.nn import functional as F
from torch.linalg import pinv

class OSELM(nn.Module):
    
    def __init__(self, activation_func, n_input_nodes, n_hidden_nodes):
        super().__init__()

        self.__n_input_nodes = n_input_nodes
        self.__n_hidden_nodes = n_hidden_nodes

        if activation_func == "sigmoid":
            self.__activation_func = F.sigmoid
        else:
            raise ValueError("Activation function not supported")

        # Initialize the input weights and bias
        self.__alpha = nn.Parameter(torch.randn(n_input_nodes, n_hidden_nodes))
        self.__bias = nn.Parameter(torch.randn(n_hidden_nodes))

        self.__p = nn.Parameter(torch.zeros(n_hidden_nodes, n_hidden_nodes))
        self.__beta = nn.Parameter(torch.zeros(n_hidden_nodes, n_input_nodes))
        self.__u = nn.Parameter(torch.zeros(n_hidden_nodes, n_hidden_nodes))
        self.__v = nn.Parameter(torch.zeros(n_hidden_nodes, n_input_nodes))

    def init_phase(self, input_matrix, target_matrix):
        H = self.__activation_func(torch.matmul(input_matrix, self.__alpha) + self.__bias)
        H_T = torch.transpose(H, 0, 1)
        T = torch.transpose(target_matrix, 0, 1)
        self.__p = pinv(torch.matmul(H_T, H))
        self.__beta = torch.matmul(torch.matmul(self.__p, H_T), T)
        return self.__beta

    # TODO Get rid of target data
    def seq_phase(self, input_data, target_data, mode="chunk"):
        H = self.__activation_func(torch.matmul(input_data, self.__alpha) + self.__bias)
        H_T = torch.transpose(H, 0, 1)
        if mode == "chunk":
            chunk_size = input_data.shape[0]
            self.calc_p_chunk(chunk_size, H, H_T)
            self.calc_beta_chunk(target_data, H, H_T)
        elif mode == "sample":
            self.calc_p_sample(H, H_T)
            self.calc_beta_sample(target_data, H, H_T)

        return self.__beta
    
    # Calculate the p of the network based on chunk of input data
    def calc_p_chunk(self, chunk_size, H, H_T):
        PH_T = torch.matmul(self.__p, H_T)
        I = torch.eye(chunk_size)
        HPH_T_Inv = pinv(torch.matmul(H, torch.matmul(self.__p, H_T)) + I)
        HP = torch.matmul(H, self.__p)
        self.__p -= torch.matmul(torch.matmul(PH_T, HPH_T_Inv), HP)

    # Calculate the beta of the network based on chunk of input data
    def calc_beta_chunk(self, target_chunk, H, H_T):
        T = torch.transpose(target_chunk, 0, 1)
        THB = T - torch.matmul(H, self.__beta)
        self.__beta += torch.matmul(torch.matmul(self.__p, H_T), THB)

    # Calculate the p of the network based on sample of input data
    def calc_p_sample(self, H, H_T):
        H_TPH = torch.matmul(H_T, torch.matmul(self.__p, H))
        PHH_TP = torch.matmul(torch.matmul(torch.matmul(self.__p, H), H_T), self.__p)
        self.__p -= torch.div(PHH_TP, 1 + H_TPH)

    # Calculate the beta of the network based on sample of input data
    def calc_beta_sample(self, target_sample, H, H_T):
        T = torch.transpose(target_sample, 0, 1)
        THB = T - torch.matmul(H_T, self.__beta)
        self.__beta += torch.matmul(torch.matmul(self.__p, H), THB)

    def calc_interm_u(self):
        self.__u = pinv(self.__p)

    def calc_interm_v(self):
        self.__v = torch.matmul(self.__u, self.__beta)

    def integrate(self, u, v):
        self.__u += u
        self.__v += v
        self.__p = pinv(self.__u)
        self.__beta = torch.matmul(pinv(self.__u), self.__v)

    # TODO Need to check if this is correct
    def gram_schmidt_orthonormalization(self):
        proj_beta_onto_alpha = torch.matmul(self.__beta.t(), self.__alpha) / torch.matmul(self.__alpha.t(), self.__alpha) * self.__alpha
        self.__beta -= proj_beta_onto_alpha

    @property
    def input_shape(self):
        return (self.__n_input_nodes,)

    @property
    def hidden_shape(self):
        return (self.__n_hidden_nodes,)
