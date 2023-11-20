import torch
from torchvision import datasets, transforms
import torch.nn as nn
from torch.backends import mps
from torch import cuda
from torch.utils.data import random_split
from autoencoder import Autoencoder
import logging
from sys import argv
import time
import matplotlib.pyplot as plt

BATCH_SIZE = 64
TRAIN_SIZE_PROP=0.8
NUM_EPOCHS = 50
DEVICE = (
    "cuda"
    if cuda.is_available()
    else "mps"
    if mps.is_available()
    else "cpu"
)

"""
Load the data
"""
def load_data(dataset):
    logging.info(f"Loading and preparing data...")
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

    train_size = int(TRAIN_SIZE_PROP * len(data))
    test_size = len(data) - train_size
    train_data, test_data = random_split(data, [train_size, test_size])
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size = BATCH_SIZE, shuffle = True)
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size = BATCH_SIZE, shuffle = True)
    logging.info(f"Loading and preparing data complete.")
    return train_loader, test_loader, input_nodes, hidden_nodes

"""
Train the autoencoder model
:param model: The autoencoder model
"""
def train_model(dataset, model, data_loader):
    logging.info(f"Training the autoencoder model...")
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    # Time and CUDA memory tracking
    start_time = time.time()
    initial_memory = torch.cuda.memory_allocated()
    peak_memory = torch.cuda.max_memory_allocated()

    logging.info(f"Training on {len(data_loader)} batches...")
    losses = []
    times = []
    for epoch in range(NUM_EPOCHS):
        loss = 0
        epoch_start_time = time.time()
        for (img, _) in data_loader:
            img = img.reshape(-1, model.input_shape[0]).to(DEVICE)
            recon = model(img)
            train_loss = criterion(recon, img)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            loss += train_loss.item()

        epoch_end_time = time.time()
        loss /= len(data_loader)
        losses.append(loss)
        times.append(epoch_end_time - epoch_start_time)
        print(f"Epoch: {epoch+1}/{NUM_EPOCHS}, Loss: {loss:.5f}")

    end_time = time.time()
    training_time = end_time - start_time
    final_memory = torch.cuda.memory_allocated()
    peak_memory = torch.cuda.max_memory_allocated()
    title = "Training Benchmarks"
    print(f"\n{title}")
    print("=" * len(title))
    print(f"Peak memory allocated during training: {peak_memory / (1024 ** 2):.2f} MB")
    print(f"Memory used during training: {(final_memory - initial_memory) / (1024 ** 2):.2f} MB")
    print(f"Training complete. Time taken: {training_time:.2f} seconds.\n")

    # Create plots
    create_plots(dataset, losses, times)
    logging.info(f"Training complete.")

"""
Test the autoencoder model
:param model: The autoencoder model
"""
def test_model(dataset, model, data_loader):
    logging.info(f"Testing the autoencoder model...")
    criterion = nn.MSELoss()
    total_loss = 0
    logging.info(f"Testing on {len(data_loader)} batches...")
    saved_img = False
    with torch.no_grad():
        for (img, _) in data_loader:
            img = img.reshape(-1, model.input_shape[0]).to(DEVICE)
            recon = model(img)
            loss = criterion(recon, img)
            total_loss += loss.item()
            if not saved_img:
                visualize_comparisons(dataset, img.cpu().numpy(), recon.cpu().numpy())
                saved_img = True

    title = "Total Loss"
    print(f"\n{title}")
    print("=" * len(title))
    print(f'Loss: {total_loss/len(data_loader):.5f}\n')
    logging.info(f"Testing complete.")

"""
Get the dataset to use (either "mnist", "fashion-mnist", "cifar10" or "cifar100")
"""
def get_dataset():
    if len(argv) < 2:
        # Default to MNIST
        return 'mnist'
    elif argv[1] in ['mnist', 'fashion-mnist', 'cifar10', 'cifar100']:
        return argv[1]
    else:
        exit_with_usage()

"""
Exit with usage message
"""
def exit_with_usage():
    print(f"Usage: python train-autoencoder.py <dataset>")
    print("dataset: mnist, fashion-mnist, cifar10, cifar100")
    exit(1)

"""
Create plots for the loss vs epoch, time vs epoch and loss vs time
"""
def create_plots(dataset, losses, times):
    # Plot the loss vs epoch
    plt.plot(losses)
    plt.title("Loss vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(f"plots/{dataset}-loss-vs-epoch.png")

    plt.clf()

    # Plot the time vs epoch
    plt.plot(times)
    plt.title("Time vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Time (s)")
    plt.savefig(f"plots/{dataset}-time-vs-epoch.png")

    plt.clf()

    # Plot the loss vs time
    plt.plot(losses, times)
    plt.title("Loss vs Time")
    plt.xlabel("Loss")
    plt.ylabel("Time (s)")
    plt.savefig(f"plots/{dataset}-loss-vs-time.png")

"""
Visualize the original and reconstructed images
:param originals: The original images
:param reconstructions: The reconstructed images
:param dataset: The dataset used
:param n: The number of images to visualize
"""
def visualize_comparisons(dataset, originals, reconstructions, n=5):
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

    plt.savefig(f"autoencoder/results/{dataset}-reconstructions.png")


def main():
    logging.basicConfig(level=logging.INFO)
    dataset = get_dataset()
    train_loader, test_loader, input_nodes, hidden_nodes = load_data(dataset)
    logging.info("Initializing the autoencoder model...")
    model = Autoencoder(input_nodes, hidden_nodes).to(DEVICE)
    logging.info("Initializing complete.")
    train_model(dataset, model, train_loader)
    test_model(dataset, model, test_loader)

if __name__ == "__main__":
    main()
