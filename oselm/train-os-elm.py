import torch
from torch.nn import MSELoss
from oselm import OSELM
from torchvision import datasets, transforms
from torch.backends import mps
from torch import cuda
from torch.utils.data import random_split

TRAIN_SIZE_PROP = 0.6
SEQ_SIZE_PROP = 0.2
TEST_SIZE_PROP = 0.2
BATCH_SIZE = 64

# Get CPU, GPU or MPS Device for training
device = (
    "cuda"
    if cuda.is_available()
    else "mps"
    if mps.is_available()
    else "cpu"
)

def oselm_init():
    activation_func = 'sigmoid'
    n_hidden_nodes = 128
    n_input_nodes = 784
    return OSELM(n_input_nodes, n_hidden_nodes, activation_func).to(device)

# Data Loading and Splitting
def load_and_split_data():
    transform = transforms.ToTensor()
    mnist_data = datasets.MNIST(root = './data', train = True, download = True, transform = transform)
    # TODO Check max between train_size and n_hidden_nodes
    # train_size *must* be greater than n_hidden_nodes
    train_size = int(0.6 * len(mnist_data))
    seq_size = int(0.2 * len(mnist_data))
    test_size = len(mnist_data) - train_size - seq_size
    train_data, seq_data, test_data = random_split(mnist_data, [train_size, seq_size, test_size])
    return train_data, seq_data, test_data

# Data Loaders
def create_data_loaders(train_data, seq_data, test_data):
    train_loader = torch.utils.data.DataLoader(train_data, batch_size = BATCH_SIZE, shuffle = True)
    seq_loader = torch.utils.data.DataLoader(seq_data, batch_size = BATCH_SIZE, shuffle = True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size = BATCH_SIZE, shuffle = True)
    return train_loader, seq_loader, test_loader

def train_model(model, data_loader, mode="sample"):
    model.train()
    criterion = MSELoss()
    total_loss = 0

def main():
    train_data, seq_data, test_data = load_and_split_data()
    train_loader, seq_loader, test_loader = create_data_loaders(train_data, seq_data, test_data)
    model = oselm_init()

    # Training
    model.init_phase()

if __name__ == "__main__":
    main()
