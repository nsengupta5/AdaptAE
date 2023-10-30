import torch
from torch import nn
from torch.linalg import pinv

class OSELM(nn.Module):
    
    def __init__(self, activation_func, loss_func, n_input_nodes, n_hidden_nodes):
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

        # Initialize the input weights and bias and make them orthogonal
        self.__alpha = nn.Parameter(torch.randn(n_input_nodes, n_hidden_nodes))
        self.__bias = nn.Parameter(torch.randn(n_hidden_nodes))

        self.__p = nn.Parameter(torch.zeros(n_hidden_nodes, n_hidden_nodes)).clone().detach().to('cuda')
        self.__beta = nn.Parameter(torch.zeros(n_hidden_nodes, n_input_nodes)).clone().detach().to('cuda')
        self.__u = nn.Parameter(torch.zeros(n_hidden_nodes, n_hidden_nodes))
        self.__v = nn.Parameter(torch.zeros(n_hidden_nodes, n_input_nodes))

    def predict(self, test_data):
        H = self.__activation_func(torch.matmul(test_data, self.__alpha) + self.__bias)
        return torch.matmul(H, self.__beta)

    def evaluate(self, test_data, pred_data):
        loss = self.__loss_func(self.predict(test_data), pred_data)
        accuracy = torch.sum(torch.argmax(self.predict(test_data), dim=1) == torch.argmax(pred_data, dim=1)) / len(pred_data) * 100
        return loss, accuracy

    def init_phase(self, data):
        H = self.__activation_func(torch.matmul(data, self.__alpha) + self.__bias)
        H_T = torch.transpose(H, 0, 1)
        self.__p = pinv(torch.matmul(H_T, H))
        pH_T = torch.matmul(self.__p, H_T)
        self.__beta = torch.matmul(pH_T, data)
        return self.__beta

    def seq_phase(self, data, mode):
        H = torch.transpose(self.__activation_func(torch.matmul(data, self.__alpha) + self.__bias), 0, 1)
        H_T = torch.transpose(H, 0, 1)
        if mode == "chunk":
            chunk_size = data.shape[0]
            self.calc_p_chunk(chunk_size, H, H_T)
            self.calc_beta_chunk(data, H, H_T)
        elif mode == "sample":
            self.calc_p_sample(H, H_T)
            self.calc_beta_sample(data, H, H_T)
        else:
            raise ValueError("Mode not supported")

        return self.__beta
    
    # Calculate the p of the network based on chunk of input data
    def calc_p_chunk(self, chunk_size, H, H_T):
        PH_T = torch.matmul(self.__p, H_T)
        I = torch.eye(chunk_size)
        HPH_T_Inv = pinv(torch.matmul(H, torch.matmul(self.__p, H_T)) + I)
        HP = torch.matmul(H, self.__p)
        self.__p -= torch.matmul(torch.matmul(PH_T, HPH_T_Inv), HP)

    # Calculate the beta of the network based on chunk of input data
    def calc_beta_chunk(self, chunk, H, H_T):
        T = torch.transpose(chunk, 0, 1)
        THB = T - torch.matmul(H, self.__beta)
        self.__beta += torch.matmul(torch.matmul(self.__p, H_T), THB)

    # Calculate the p of the network based on sample of input data
    def calc_p_sample(self, H, H_T):
        PHH_TP = torch.matmul(torch.matmul(torch.matmul(self.__p, H), H_T), self.__p)
        H_TPH = torch.matmul(H_T, torch.matmul(self.__p, H))
        self.__p -= torch.div(PHH_TP, 1 + H_TPH)

    # Calculate the beta of the network based on sample of input data
    def calc_beta_sample(self, sample, H, H_T):
        THB = sample - torch.matmul(H_T, self.__beta)
        PH = torch.matmul(self.__p, H)
        self.__beta += torch.matmul(PH, THB)

    def calc_interm_u(self):
        self.__u = pinv(self.__p)

    def calc_interm_v(self):
        self.__v = torch.matmul(self.__u, self.__beta)

    def integrate(self, u, v):
        self.__u += u
        self.__v += v
        self.__p = pinv(self.__u)
        self.__beta = torch.matmul(pinv(self.__u), self.__v)

    def gram_schmidt(self,factor):
        Q, _ = torch.linalg.qr(factor)
        return Q

    @property
    def input_shape(self):
        return (self.__n_input_nodes,)

    @property
    def hidden_shape(self):
        return (self.__n_hidden_nodes,)
