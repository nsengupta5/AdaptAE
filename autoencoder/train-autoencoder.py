from autoencoder import Autoencoder
import torch
from torchvision import datasets, transforms
import torch.nn as nn
import logging
from sys import argv
import time
import matplotlib.pyplot as plt
import psutil

BATCH_SIZE = 64
NUM_EPOCHS = 30
NUM_IMAGES = 5
DEBUG = True

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
            train_data = datasets.MNIST(root = './data', train = True, download = True, transform = transform)
            test_data = datasets.MNIST(root = './data', train = False, download = True, transform = transform)
        case 'fashion-mnist':
            transform = transforms.ToTensor()
            train_data = datasets.FashionMNIST(root = './data', train = True, download = True, transform = transform)
            test_data = datasets.FashionMNIST(root = './data', train = False, download = True, transform = transform)
        case 'cifar10':
            transform = transforms.Compose([
                transforms.ToTensor(),
                # Normalize each channel of the input data
                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
            ])
            train_data = datasets.CIFAR10(root = './data', train = True, download = True, transform = transform)
            test_data = datasets.CIFAR10(root = './data', train = False, download = True, transform = transform)
            input_nodes = 3072
            hidden_nodes = 1024
        case 'cifar100':
            transform = transforms.Compose([
                transforms.ToTensor(),
                # Normalize each channel of the input data
                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
            ])
            train_data = datasets.CIFAR100(root = './data', train = True, download = True, transform = transform)
            test_data = datasets.CIFAR100(root = './data', train = False, download = True, transform = transform)
            input_nodes = 3072
            hidden_nodes = 1024
        case 'super-tiny-imagenet':
            transform = transforms.Compose([
                transforms.Resize((32,32)),
                transforms.ToTensor(),
                # Normalize each channel of the input data
                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
            ])
            train_data = datasets.ImageFolder(root = './data/tiny-imagenet-200/train', transform = transform)
            test_data = datasets.ImageFolder(root = './data/tiny-imagenet-200/test', transform = transform)
            input_nodes = 3072
            hidden_nodes = 1024
        case 'tiny-imagenet':
            transform = transforms.Compose([
                transforms.Resize((64,64)),
                transforms.ToTensor(),
                # Normalize each channel of the input data
                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
            ])
            train_data = datasets.ImageFolder(root = './data/tiny-imagenet-200/train', transform = transform)
            test_data = datasets.ImageFolder(root = './data/tiny-imagenet-200/test', transform = transform)
            input_nodes = 12288
            hidden_nodes = 4096
        case _:
            raise ValueError(f"Invalid dataset: {dataset}")

    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size = BATCH_SIZE, shuffle = True)
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size = BATCH_SIZE, shuffle = False)
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
    peak_memory = 0
    process = None
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    elif device == 'cpu':
        process = psutil.Process()

    logging.info(f"Training on {len(data_loader)} batches...")
    losses = []
    times = []
    start_time = time.time()
    for epoch in range(NUM_EPOCHS):
        loss = 0
        epoch_start_time = time.time()
        for (img, _) in data_loader:
            img = img.reshape(-1, model.input_shape[0]).to(device)
            recon = model(img)
            train_loss = criterion(recon, img)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            loss += train_loss.item()

        epoch_end_time = time.time()
        loss /= len(data_loader)

        if device == "cpu":
            curr_memory = process.memory_info().rss
            peak_memory = max(peak_memory, curr_memory)

        losses.append(loss)
        times.append(epoch_end_time - epoch_start_time)
        print(f"Epoch: {epoch+1}/{NUM_EPOCHS}, Loss: {loss:.5f}")

    end_time = time.time()
    training_time = end_time - start_time

    if device == 'cuda':
        peak_memory = torch.cuda.max_memory_allocated()

    title = "Training Benchmarks"
    print(f"\n{title}")
    print("=" * len(title))
    print(f"Peak memory allocated during training: {peak_memory / (1024 ** 2):.2f} MB")
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
            img = img.reshape(-1, model.input_shape[0]).to(device)
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
Get the dataset to use (either "mnist", "fashion-mnist", "cifar10", "cifar100", "tiny-imagenet")
"""
def get_dataset():
    if len(argv) < 2:
        # Default to MNIST
        return 'mnist'
    elif argv[1] in ['mnist', 'fashion-mnist', 'cifar10', 'cifar100', 'super-tiny-imagenet' ,'tiny-imagenet']:
        return argv[1]
    else:
        exit_with_usage()

"""
Get the device to use (either "cpu", "mpu" or "cuda")
"""
def get_device():
    if len(argv) < 3:
        # Default to CPU
        return "cuda"
    elif argv[2] not in ["cpu", "mpu", "cuda"]:
        exit_with_usage()
    else:
        return argv[2]

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
def visualize_comparisons(dataset, originals, reconstructions, n=NUM_IMAGES):
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # Display original images
        ax = plt.subplot(2, n, i + 1)
        if dataset in ["mnist", "fashion-mnist"]:
            plt.imshow(originals[i].reshape(28, 28))
        elif dataset in ["cifar10", "cifar100", "super-tiny-imagenet"]:
            plt.imshow(originals[i].reshape(3, 32, 32).transpose(1, 2, 0))
        else:
            plt.imshow(originals[i].reshape(3, 64, 64).transpose(1, 2, 0))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Display reconstructed images
        ax = plt.subplot(2, n, i + 1 + n)
        if dataset in ["mnist", "fashion-mnist"]:
            plt.imshow(reconstructions[i].reshape(28, 28))
        elif dataset in ["cifar10", "cifar100", "super-tiny-imagenet"]:
            plt.imshow(reconstructions[i].reshape(3, 32, 32).transpose(1, 2, 0))
        else:
            plt.imshow(reconstructions[i].reshape(3, 64, 64).transpose(1, 2, 0))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.savefig(f"autoencoder/results/{dataset}-reconstructions.png")


def main():
    logging.basicConfig(level=logging.INFO)
    global device
    device = get_device()
    dataset = get_dataset()
    train_loader, test_loader, input_nodes, hidden_nodes = load_data(dataset)
    logging.info("Initializing the autoencoder model...")
    model = Autoencoder(input_nodes, hidden_nodes).to(device)
    logging.info("Initializing complete.")
    train_model(dataset, model, train_loader)
    test_model(dataset, model, test_loader)

if __name__ == "__main__":
    main()
