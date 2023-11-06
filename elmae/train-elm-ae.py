from elmae import ELMAE, assert_cond
from torchvision import datasets, transforms
from torch.backends import mps
from torch import cuda, clamp, set_printoptions
from torch.utils.data import random_split
from sys import argv
import logging

# Constants
TRAIN_SIZE_PROP = 0.8
TEST_SIZE_PROP = 0.2
DEVICE = (
    "cuda"
    if cuda.is_available()
    else "mps"
    if mps.is_available()
    else "cpu"
)

"""
Initialize the ELMAE model
"""
def elmae_init(input_nodes, hidden_nodes):
    logging.info(f"Initializing ELMAE model...")
    activation_func = 'sigmoid'
    loss_func = 'mse'
    model = ELMAE(activation_func, loss_func, input_nodes, hidden_nodes, DEVICE).to(DEVICE)
    # Orthogonalize the hidden parameters
    logging.info(f"Orthogonalizing hidden parameters...")
    model.orthogonalize_hidden_params()
    logging.info(f"Orthogonalizing hidden parameters complete.")
    logging.info(f"Initializing ELMAE model complete.")
    return model

"""
Load and split the data into training, sequential and test data
"""
def load_and_split_data(dataset):
    logging.info(f"Loading and preparing data...")
    transform = transforms.ToTensor()
    input_nodes = 784
    hidden_nodes = 128
    match dataset:
        case 'mnist':
            transform = transforms.ToTensor()
            data = datasets.MNIST(root = './data', train = True, download = True, transform = transform)
        case 'fashion-mnist':
            transform = transforms.ToTensor()
            data = datasets.FashionMNIST(root = './data', train = True, download = True, transform = transform)
        case 'cifar10':
            transform = transforms.Compose([
                transforms.ToTensor(),
                # Normalize each channel of the input data
                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
            ])
            data = datasets.CIFAR10(root = './data', train = True, download = True, transform = transform)
            input_nodes = 1024
            hidden_nodes = 512
        case 'cifar100':
            transform = transforms.Compose([
                transforms.ToTensor(),
                # Normalize each channel of the input data
                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
            ])
            data = datasets.CIFAR100(root = './data', train = True, download = True, transform = transform)
        case _:
            raise ValueError(f"Invalid dataset: {dataset}")

    # Split 80% for training and 20% for testing
    train_size = int(TRAIN_SIZE_PROP * len(data))
    test_size = len(data) - train_size
    train_data, test_data = random_split(data, [train_size, test_size])
    logging.info(f"Loading and preparing data complete.")
    return train_data, test_data, input_nodes, hidden_nodes

"""
Train the model
:param model: The ELMAE model to train
:param train_data: The training data
"""
def train_model(model, train_data):
    # Assert that the initial training data is of the correct shape
    data = train_data.dataset.data.view(-1, model.input_shape[0]).float().to(DEVICE) / 255
    assert_cond(data.shape[0] == len(train_data.dataset), "Train data shape mismatch")
    logging.info(f"Training on {len(data)} samples...")
    logging.info("Train data shape: " + str(data.shape))
    model.calc_beta_sparse(data)
    logging.info(f"Training complete.")
    
"""
Test the model
:param model: The ELMAE model to test
:param test_data: The test data
"""
def test_model(model, test_data):
    logging.info(f"Testing on {len(test_data.dataset)} samples...")
    set_printoptions(sci_mode=False)
    data = test_data.dataset.data.view(-1, model.input_shape[0]).float().to(DEVICE) / 255
    assert_cond(data.shape[0] == len(test_data.dataset), "Test data shape mismatch")
    pred = model.predict(data)
    pred = clamp(pred, min=0).round().int()
    loss, _ = model.evaluate(data, pred)
    print(f"Loss: {loss.item():.5f}")
    logging.info(f"Testing complete.")

"""
Exit the program with an error message of the correct usage
"""
def exit_with_error():
    print("Usage: python train-elm-ae.py [dataset]")
    print("dataset: mnist, fashion-mnist, cifar10 or cifar100")
    exit(1)

"""
Get the dataset to use (either "mnist", "fashion-mnist", "cifar10" or "cifar100")
"""
def get_dataset():
    if len(argv) < 2:
        # Default to MNIST datasets
        return "mnist"
    elif argv[1] not in ["mnist", "fashion-mnist", "cifar10", "cifar100"]:
        exit_with_error()
    else:
        return argv[1]

def main():
    dataset = get_dataset()
    logging.basicConfig(level=logging.INFO)
    train_data, test_data, input_nodes, hidden_nodes = load_and_split_data(dataset)
    model = elmae_init(input_nodes, hidden_nodes)
    train_model(model, train_data)
    test_model(model, test_data)

if __name__ == "__main__":
    main()
