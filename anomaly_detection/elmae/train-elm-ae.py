"""
File: train--elm-ae.py
Author: Nikhil Sengupta
Created on: November 6, 2023
Last Modified: December 12, 2023
Email: ns214@st-andrews.ac.uk

Description: 
    This file contains 

License:
    This code is released under the MIT License

Usage:
    python train-elm-ae.py [-h] --dataset {mnist,fashion-mnist,cifar10,cifar100,
                           super-tiny-imagenet,tiny-imagenet} 
                           [--device {cpu,mps,cuda}] [--num-images NUM_IMAGES] 
                           [--generate-imgs] [--save-results]


    options:
      -h, --help            show this help message and exit

      --dataset {mnist,fashion-mnist,cifar10,cifar100,super-tiny-imagenet,tiny-imagenet}
                            The dataset to use 
                            (either 'mnist', 'fashion-mnist', 'cifar10', 'cifar100', 
                            'super-tiny-imagenet' or 'tiny-imagenet')

      --device {cpu,mps,cuda}
                            The device to use (either 'cpu', 'mps' or 'cuda'). 
                            Defaults to 'cuda' if not provided

      --num-images NUM_IMAGES
                            The number of images to generate. 
                            Defaults to 5 if not provided

      --generate-imgs       Whether to generate images of the reconstructions

      --save-results        Whether to save the results to a CSV file

Example: python train-elm-ae.py --dataset cifar10 
"""

from models.elmae import ELMAE
from util.data import *
from util.util import *
import torch
import logging
import time
import warnings
import psutil
import argparse

# Constants
DEFAULT_NUM_IMAGES = 5
result_data = []

"""
Initialize the ELMAE model
:param input_nodes: The number of input nodes
:type input_nodes: int
:param hidden_nodes: The number of hidden nodes
:type hidden_nodes: int
:return model: The initialized ELMAE model
:rtype model: ELMAE
"""
def elmae_init(input_nodes, hidden_nodes):
    logging.info(f"Initializing ELMAE model...")
    activation_func = 'sigmoid'
    loss_func = 'mse'
    model = ELMAE(activation_func, loss_func, input_nodes, hidden_nodes, device).to(device)
    logging.info(f"Initializing ELM-AE model complete.\n")
    return model

"""
Load and split the data
:param dataset: The dataset to load
:type dataset: str
:return train_loader: The training data
:rtype train_loader: torch.utils.data.DataLoader
:return test_loader: The test data
:rtype test_loader: torch.utils.data.DataLoader
:return input_nodes: The number of input nodes
:rtype input_nodes: int
:return hidden_nodes: The number of hidden nodes
:rtype hidden_nodes: int
"""
def load_and_split_data(dataset):
    logging.info(f"Loading and preparing data...")

    # Load the data
    input_nodes, hidden_nodes, train_data, test_data = load_data(dataset)

    train_size = len(train_data)
    test_size = len(test_data)

    # Create the data loaders
    train_loader = torch.utils.data.DataLoader(
        Loader(train_data),
        batch_size=train_size,
        shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        Loader(test_data),
        batch_size=test_size,
        shuffle=False
    )

    logging.info(f"Loading and preparing data complete.")
    return train_loader, test_loader, input_nodes, hidden_nodes

"""
Train the model
:param model: The ELMAE model to train
:type model: ELMAE
:param train_loader: The training data
:type train_loader: torch.utils.data.DataLoader
"""
def train_model(model, train_loader):
    peak_memory = 0
    process = None

    for _, (data) in enumerate(train_loader):
        # Reshape the data
        data = data.to(device)
        logging.info(f"Training on {len(data)} samples...")

        # Reset peak memory stats
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        else:
            process = psutil.Process()
            peak_memory = process.memory_info().rss

        # Start time tracking
        start_time = time.time()

        model.calc_beta(data)

        # End time tracking
        end_time = time.time()

        # Final memory usage
        if device == "cuda":
            peak_memory = torch.cuda.max_memory_allocated()
        else:
            current_memory = process.memory_info().rss
            peak_memory = max(peak_memory, current_memory)

        # Calculate time taken and memory used
        time_taken = end_time - start_time

        # Evaluate the model
        pred = model.predict(data)
        loss, _ = evaluate(data, pred)
        
        # Print results
        print_header("Training Benchmarks")
        peak_memory = peak_memory / (1024 ** 2)
        print(f"Peak memory allocated during training: {peak_memory:.2f} MB")
        print(f"Time taken: {time_taken:.2f} seconds.")
        print(f"Loss: {loss.item():.5f}")

        # Save results
        result_data.append(peak_memory)
        result_data.append(time_taken)
        result_data.append(loss.item())

        logging.info(f"Training complete.\n")

"""
Test the model
:param model: The ELMAE model to test
:type model: ELMAE
:param test_data: The test data
:type test_data: torch.utils.data.DataLoader
:param generate_imgs: Whether to generate images of the reconstructions
:type generate_imgs: bool
:param num_imgs: The number of images to generate
:type num_imgs: int
"""
def test_model(model, test_loader, dataset, generate_imgs, num_imgs):
    for _, (data) in enumerate(test_loader):
        # Reshape the data
        data = data.to(device)
        logging.info(f"Testing on {len(data)} samples...")
        pred = model.predict(data)

        # Visualize the results
        if generate_imgs:
            results_file = f"elmae/results/{dataset}-reconstructions.png"
            visualize_comparisons(
                data.cpu().numpy(),
                pred.cpu().detach().numpy(),
                dataset,
                num_imgs,
                results_file
            )

        loss, _ = evaluate(data, pred)

        # Print results
        print_header("Total Loss")
        print(f"Loss: {loss.item():.5f}")

        # Save results
        result_data.append(loss.item())

        logging.info(f"Testing complete.\n")

"""
Get the arguments from the command line
"""
def get_args():
    parser = argparse.ArgumentParser(description="Train an ELM-AE model")
    # Define the arguments
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
        "--num-images",
        type=int,
        default=DEFAULT_NUM_IMAGES,
        help="The number of images to generate. Defaults to 5 if not provided"
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

    # Parse the arguments
    args = parser.parse_args()
    dataset = args.dataset
    device = args.device
    num_images = args.num_images
    generate_imgs = args.generate_imgs
    save_results = args.save_results

    return dataset, device, num_images, generate_imgs, save_results

def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    logging.basicConfig(level=logging.INFO)

    # Get the arguments
    global device
    dataset, device, num_images, generate_imgs, save_results = get_args()

    train_loader, test_loader, input_nodes, hidden_nodes = load_and_split_data(dataset)
    model = elmae_init(input_nodes, hidden_nodes)
    train_model(model, train_loader)
    test_model(model, test_loader, dataset, generate_imgs, num_images)

    if save_results:
        save_result_data("elmae", dataset, None, None, result_data)
        plot_latent_representation(model, test_loader, f"elmae/plots/latents/{dataset}-latent-representation.png")

if __name__ == "__main__":
    main()
