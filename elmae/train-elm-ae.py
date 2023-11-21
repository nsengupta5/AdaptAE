from elmae import ELMAE
from torchvision import datasets, transforms
from torch.backends import mps
from torch import cuda
from torch.utils.data import random_split
from sys import argv
import torch
import logging
import time
import warnings
import matplotlib.pyplot as plt

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
    # logging.info(f"Orthogonalizing hidden parameters...")
    # model.orthogonalize_hidden_params()
    # logging.info(f"Orthogonalizing hidden parameters complete.")
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
            input_nodes = 3072
            hidden_nodes = 1024
        case 'cifar100':
            transform = transforms.Compose([
                transforms.ToTensor(),
                # Normalize each channel of the input data
                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
            ])
            data = datasets.CIFAR100(root = './data', train = True, download = True, transform = transform)
            input_nodes = 3072
            hidden_nodes = 1024
        case _:
            raise ValueError(f"Invalid dataset: {dataset}")

    # Split 80% for training and 20% for testing
    train_size = int(TRAIN_SIZE_PROP * len(data))
    test_size = len(data) - train_size
    train_data, test_data = random_split(data, [train_size, test_size])
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size = train_size, shuffle = True)
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size = test_size, shuffle = True)
    logging.info(f"Loading and preparing data complete.")
    return train_loader, test_loader, input_nodes, hidden_nodes

"""
Train the model
:param model: The ELMAE model to train
:param train_data: The training data
"""
def train_model(model, train_loader):
    for (data, _) in train_loader:
        # Reshape the data
        data = data.reshape(-1, model.input_shape[0]).float().to(DEVICE)
        logging.info(f"Training on {len(data)} samples...")
        logging.info("Train data shape: " + str(data.shape))

        # Start time tracking
        start_time = time.time()
        # Initial memory usage
        initial_memory = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()

        # Train the model
        model.calc_beta_sparse(data)

        # End time tracking
        end_time = time.time()

        # Final memory usage
        final_memory = torch.cuda.memory_allocated()
        peak_memory = torch.cuda.max_memory_allocated()

        # Calculate time taken and memory used
        time_taken = end_time - start_time
        memory_used = final_memory - initial_memory
        
        title = "Training Benchmarks:"
        print(f"\n{title}")
        print("=" * len(title))
        print(f"Peak memory allocated during training: {peak_memory / (1024 ** 2):.2f} MB")
        print(f"Memory used during training: {memory_used / (1024 ** 2):.2f} MB")
        print(f"Training complete. Time taken: {time_taken:.2f} seconds.")

        # Evaluate the model
        pred = model.predict(data)
        loss, _ = model.evaluate(data, pred)
        print(f"Loss: {loss.item():.5f}\n")

        logging.info(f"Training complete.")

"""
Test the model
:param model: The ELMAE model to test
:param test_data: The test data
"""
def test_model(model, test_loader, dataset):
    for (data, _) in test_loader:
        # Reshape the data
        data = data.reshape(-1, model.input_shape[0]).float().to(DEVICE)
        logging.info(f"Testing on {len(data)} samples...")
        logging.info("Test data shape: " + str(data.shape))
        pred = model.predict(data)

        # Visualize the results
        visualize_comparisons(data.cpu().numpy(), pred.cpu().detach().numpy(), dataset)

        loss, _ = model.evaluate(data, pred)
        title = "Total Loss:"
        print(f"\n{title}")
        print("=" * len(title))
        print(f"Loss: {loss.item():.5f}\n")
        logging.info(f"Testing complete.")

"""
Exit the program with an error message of the correct usage
"""
def exit_with_error():
    print("Usage: python train-elm-ae.py [dataset]")
    print("dataset: mnist, fashion-mnist, cifar10 or cifar100")
    exit(1)


"""
Visualize the original and reconstructed images
:param originals: The original images
:param reconstructions: The reconstructed images
:param dataset: The dataset used
:param n: The number of images to visualize
"""
def visualize_comparisons(originals, reconstructions, dataset, n=5):
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # Display original images
        ax = plt.subplot(2, n, i + 1)
        if dataset in ["mnist", "fashion-mnist"]:
            plt.imshow(originals[i].reshape(28, 28))
        else:
            plt.imshow(originals[i].reshape(3, 32, 32).transpose(1, 2, 0))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Display reconstructed images
        ax = plt.subplot(2, n, i + 1 + n)
        if dataset in ["mnist", "fashion-mnist"]:
            plt.imshow(reconstructions[i].reshape(28, 28))
        else:
            plt.imshow(reconstructions[i].reshape(3, 32, 32).transpose(1, 2, 0))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.savefig(f"elmae/results/{dataset}-reconstructions.png")

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
    warnings.filterwarnings("ignore", category=UserWarning)
    dataset = get_dataset()
    logging.basicConfig(level=logging.INFO)
    train_loader, test_loader, input_nodes, hidden_nodes = load_and_split_data(dataset)
    model = elmae_init(input_nodes, hidden_nodes)
    train_model(model, train_loader)
    test_model(model, test_loader, dataset)

if __name__ == "__main__":
    main()
