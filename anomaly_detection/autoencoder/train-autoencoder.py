"""
File: train-autoencoder.py
Author: Nikhil Sengupta
Created on: November 6, 2023
Last Modified: December 12, 2023
Email: ns214@st-andrews.ac.uk

Description: 
    This file contains 

License:
    This code is released under the MIT License

Usage:
    python train-autoencoder.py [-h] --dataset {mnist,fashion-mnist,cifar10,
                                cifar100,super-tiny-imagenet,tiny-imagenet} 
                                [--device {cpu,mps,cuda}] [--generate-imgs] 
                                [--save-results] [--num-images NUM_IMAGES] 
                                [--num-epochs NUM_EPOCHS] [--batch-size BATCH_SIZE]

    Train an autoencoder model

    options:
      -h, --help            show this help message and exit

      --dataset {mnist,fashion-mnist,cifar10,cifar100,super-tiny-imagenet,tiny-imagenet}
                            The dataset to use 
                            (either 'mnist', 'fashion-mnist', 'cifar10', 'cifar100', 
                            'super-tiny-imagenet' or 'tiny-imagenet')

      --device {cpu,mps,cuda}
                            The device to use (either 'cpu', 'mps' or 'cuda'). 
                            Defaults to 'cuda' if not provided

      --generate-imgs       Whether to generate images of the reconstructions

      --save-results        Whether to save the results to a CSV file

      --num-images NUM_IMAGES
                            The number of images to generate. 
                            Defaults to 5 if not provided

      --num-epochs NUM_EPOCHS
                            The number of epochs to train for. 
                            Defaults to 30 if not provided

      --batch-size BATCH_SIZE
                            The batch size to use. 
                            Defaults to 64 if not provided

Example: python train-autoencoder.py --dataset mnist --num-epochs 50
"""

from models.autoencoder import Autoencoder
from util.util import *
from util.data import *
import torch
import logging
import time
import psutil
import argparse
import seaborn as sns

# Constants
DEFAULT_BATCH_SIZE = 64
DEFAULT_NUM_EPOCHS = 30
DEFAULT_NUM_IMAGES = 5
result_data = []

"""
Initialize the Autoencoder model
:param input_nodes: The number of input nodes
:type input_nodes: int
:param hidden_nodes: The number of hidden nodes
:type hidden_nodes: int
:return: The initialized Autoencoder model
:rtype: Autoencoder
"""
def autoencoder_init(input_nodes, hidden_nodes):
    logging.info(f"Initializing Autoencoder model...")
    model = Autoencoder(input_nodes, hidden_nodes, device).to(device)
    logging.info("Initializig autoencoder Autoencoder model complete.\n")
    return model

"""
Load and split the data
:param dataset: The dataset to load
:type dataset: str
:param batch_size: The batch size
:type batch_size: int
:return train_loader: The training data loader
:rtype train_loader: torch.utils.data.DataLoader
:return test_loader: The test data loader
:rtype test_loader: torch.utils.data.DataLoader
:return input_nodes: The number of input nodes
:rtype input_nodes: int
:return hidden_nodes: The number of hidden nodes
:rtype hidden_nodes: int
"""
def load_and_split_data(dataset, batch_size):
    logging.info(f"Loading and preparing data...")

    # Load the data
    input_nodes, hidden_nodes, train_data, test_data = load_data(dataset)

    # Create the data loaders
    train_loader = torch.utils.data.DataLoader(
        Train_Loader(),
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        Test_Loader(),
        batch_size=batch_size,
        shuffle=False
    )
    
    logging.info(f"Loading and preparing data complete.")
    return train_loader, test_loader, input_nodes, hidden_nodes

"""
Train the autoencoder model
:param model: The autoencoder model
:type model: Autoencoder
:param data_loader: The training data loader
:type data_loader: torch.utils.data.DataLoader
:param num_epochs: The number of epochs to train for
:type num_epochs: int
"""
def train_model(model, data_loader, num_epochs):
    logging.info(f"Training the autoencoder model...")
    
    # Set the model to training mode
    model.train()

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
    for epoch in range(num_epochs):
        loss = 0
        epoch_start_time = time.time()
        for _, (img) in enumerate(data_loader):
            img = img.to(device)
            recon = model(img)
            train_loss, _ = evaluate(img, recon)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            loss += train_loss.item()

        epoch_end_time = time.time()
        loss /= len(data_loader)

        if device == "cpu":
            curr_memory = process.memory_info().rss
            peak_memory = max(peak_memory, curr_memory)

        # Save the loss and time for this epoch
        losses.append(loss)
        times.append(epoch_end_time - epoch_start_time)
        print(f"Epoch: {epoch+1}/{num_epochs}, Loss: {loss:.5f}")

    end_time = time.time()
    training_time = end_time - start_time

    if device == 'cuda':
        peak_memory = torch.cuda.max_memory_allocated()

    # Print the training benchmarks
    print_header("Training Benchmarks")
    print(f"Peak memory allocated during training: {peak_memory / (1024 ** 2):.2f} MB")
    print(f"Time taken: {training_time:.2f} seconds.")
    print(f"Average loss per epoch: {sum(losses) / len(losses):.5f}\n")

    logging.info(f"Training complete.\n")

"""
Test the autoencoder model
:param model: The autoencoder model
:type model: Autoencoder
:param data_loader: The test data loader
:type data_loader: torch.utils.data.DataLoader
:param dataset: The dataset
:type dataset: str
:param gen_imgs: Whether to generate the reconstructed images
:type gen_imgs: bool
:param num_imgs: The number of images to generate
:type num_imgs: int
"""
def test_model(model, data_loader, dataset, gen_imgs, num_imgs):
    logging.info(f"Testing the autoencoder model...")

    # Set the model to evaluation mode
    model.eval()

    losses = []
    logging.info(f"Testing on {len(data_loader)} batches...")
    saved_img = False
    with torch.no_grad():
        for _, (img) in enumerate(data_loader):
            # Reshape the image to fit the model
            img = img.to(device)
            recon = model(img)
            loss, _ = evaluate(img, recon)
            losses.append(loss.item())
            if gen_imgs:
                # Only save the first num_imgs images
                if not saved_img:
                    results_file = f"anomaly_detection/autoencoder/results/{dataset}-reconstructions.png"
                    visualize_comparisons(
                        img.cpu().numpy(), 
                        recon.cpu().numpy(), 
                        dataset,
                        num_imgs,
                        results_file
                    )
                    saved_img = True

    # Print results
    print_header("Test Benchmarks")
    total_loss = sum(losses)
    
    plt.figure(figsize=(12, 6))
    plt.title("Loss Distribution")
    sns.distplot(losses, bins=20, kde=False, color="blue")
    plt.xlabel("Loss")
    plt.ylabel("Count")
    plt.show()

    print(f'Loss: {total_loss/len(data_loader):.5f}\n')

    logging.info(f"Testing complete.")

"""
Get the arguments from the command line
:return dataset: The dataset to use
:rtype dataset: str
:return device: The device to use
:rtype device: str
:return generate_imgs: Whether to generate the reconstructed images
:rtype generate_imgs: bool
:return save_results: Whether to save the results
:rtype save_results: bool
:return num_imgs: The number of images to generate
:rtype num_imgs: int
:return num_epochs: The number of epochs to train for
:rtype num_epochs: int
:return batch_size: The batch size
:rtype batch_size: int
"""
def get_args():
    parser = argparse.ArgumentParser(description='Train an autoencoder model')
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["mnist-corrupted"],
        required=True,
        help=("The dataset to use (either 'mnist', 'fashion-mnist', 'cifar10', "
              "'cifar100', 'super-tiny-imagenet' or 'tiny-imagenet')")
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "mps", "cuda"],
        default="cuda",
        help=("The device to use (either 'cpu', 'mps' or 'cuda'). "
              "Defaults to 'cuda' if not provided")
    )
    parser.add_argument(
        "--generate-imgs",
        action="store_true",
        help="Whether to generate images of the reconstructions"
    )
    parser.add_argument(
        "--save-results",
        action="store_true",
        help="Whether to save the results to a CSV file"
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=DEFAULT_NUM_IMAGES,
        help="The number of images to generate. Defaults to 5 if not provided"
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=DEFAULT_NUM_EPOCHS,
        help="The number of epochs to train for. Defaults to 30 if not provided"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="The batch size to use. Defaults to 64 if not provided"
    )

    # Parse the arguments
    args = parser.parse_args()
    dataset = args.dataset
    device = args.device
    generate_imgs = args.generate_imgs
    save_results = args.save_results
    num_images = args.num_images
    num_epochs = args.num_epochs
    batch_size = args.batch_size

    return dataset, device, generate_imgs, save_results, num_images, num_epochs, batch_size 

def main():
    logging.basicConfig(level=logging.INFO)

    # Get the arguments
    global device
    dataset, device, gen_imgs, save_results, num_imgs, n_epochs, batch_size = get_args()

    train_loader, test_loader, input_nodes, hidden_nodes = load_and_split_data(dataset, batch_size)
    model = autoencoder_init(input_nodes, hidden_nodes)
    train_model(model, train_loader, n_epochs)
    test_model(model, test_loader, dataset, gen_imgs, num_imgs)

if __name__ == "__main__":
    main()