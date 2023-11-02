import torch
from torchvision import datasets, transforms
import torch.nn as nn
from torch.backends import mps
from torch import cuda
from torch.utils.data import random_split
from autoencoder import Autoencoder
import logging

NUM_EPOCHS = 1
INPUT_NODES = 784
HIDDEN_NODES = 128
DEVICE = (
    "cuda"
    if cuda.is_available()
    else "mps"
    if mps.is_available()
    else "cpu"
)

"""
Load the MNIST data
"""
def load_data():
    logging.info(f"Loading and preparing MNIST data...")
    transform = transforms.ToTensor()
    mnist_data = datasets.MNIST(root = './data', train = True, download = True, transform = transform)
    train_size = int(0.8 * len(mnist_data))
    test_size = len(mnist_data) - train_size
    train_data, test_data = random_split(mnist_data, [train_size, test_size])
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size = 64, shuffle = True)
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size = 64, shuffle = True)
    logging.info(f"Loading and preparing MNIST data complete.")
    return train_loader, test_loader

"""
Train the autoencoder model
:param model: The autoencoder model
"""
def train_model(model, data_loader):
    logging.info(f"Training the autoencoder model...")
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    for _ in range(NUM_EPOCHS):
        for (img, _) in data_loader:
            img = img.reshape(-1, 28*28).to(DEVICE)
            recon = model(img)
            loss = criterion(recon, img)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    logging.info(f"Training complete.")

def test_model(model, data_loader):
    logging.info(f"Testing the autoencoder model...")
    criterion = nn.MSELoss()
    total_loss = 0
    with torch.no_grad():
        for (img, _) in data_loader:
            img = img.reshape(-1, 28*28).to(DEVICE)
            recon = model(img)
            loss = criterion(recon, img)
            total_loss += loss.item()
    print(f'Loss: {total_loss/len(data_loader):.5f}')
    logging.info(f"Testing complete.")

def main():
    logging.basicConfig(level=logging.INFO)
    train_loader, test_loader = load_data()
    model = Autoencoder(INPUT_NODES, HIDDEN_NODES).to(DEVICE)
    train_model(model, train_loader)
    test_model(model, test_loader)

if __name__ == "__main__":
    main()
