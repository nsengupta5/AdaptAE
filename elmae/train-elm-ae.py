"""
File: train-elm-ae.py
Author: Nikhil Sengupta
Created on: November 6, 2023
Last Modified: January 5, 2024
Email: ns214@st-andrews.ac.uk

Description: 
    This file contains the implementation of the training of the ELM-AE model

License:
    This code is released under the MIT License

Usage:
    python train-elm-ae.py [-h] --dataset {mnist,fashion-mnist,cifar10,cifar100,
                           super-tiny-imagenet,tiny-imagenet} 
                           [--device {cpu,mps,cuda}] [--num-images NUM_IMAGES] 
                           [--generate-imgs] [--save-results]
                           [--result-strategy {latent,all}] 
                           --task {reconstruction,anomaly-detection}
    Train an ELM-AE model

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
                            The number of images to generate. Defaults to 5 if not provided

      --generate-imgs       Whether to generate images of the reconstructions

      --save-results        Whether to save the results to a CSV file

      --result-strategy {latent,all}
                            If saving results, the independent variable to vary when saving results

      --task {reconstruction,anomaly-detection}
                            The task to perform (either 'reconstruction' or 'anomaly-detection')

    Example: python train-elm-ae.py --dataset cifar10 --task anomaly-detection
"""

from models.elmae import ELMAE
from util.data import *
from util.util import *
import torch
from torch.utils.data import random_split
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
    activation_func = 'tanh'
    model = ELMAE(activation_func, input_nodes, hidden_nodes, device).to(device)
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
def load_and_split_data(dataset, task):
    logging.info(f"Loading and preparing data...")

    # Load the data
    input_nodes, hidden_nodes, train_data, test_data = load_data(dataset)

    # Training set reduced to 80% of original size
    # to match the quantity of the training set of
    # the autoencoder model
    train_size = int(0.8 * len(train_data))
    valid_size = len(train_data) - train_size

    train_data, _ = random_split(train_data, [train_size, valid_size])

    test_size = len(test_data)

    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size = train_size, shuffle = True)

    if task == "reconstruction":
        test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size = test_size, shuffle = False)
    else:
        # Create a noisy test loader for anomaly detection
        test_loader = torch.utils.data.DataLoader(
            NoisyLoader(test_data),
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

    for (data, _) in train_loader:
        # Reshape the data
        data = data.reshape(-1, model.input_shape[0]).float().to(device)
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
        result_data.append(time_taken)
        result_data.append(round(peak_memory, 2))
        result_data.append(float(str(f"{loss.item():3f}")))

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
def test_model(model, test_loader, dataset, generate_imgs, num_imgs, task):

    for (data, _) in test_loader:
        # Reshape the data
        data = data.reshape(-1, model.input_shape[0]).float().to(device)
        logging.info(f"Testing on {len(data)} samples...")
        pred = model.predict(data)

        # Visualize the results
        if generate_imgs:
            results_file = (
                f"elmae/results/reconstruction/{dataset}-{task}.png"
                if task == "reconstruction"
                else f"elmae/results/anomaly_detection/{dataset}-{task}.png"
            )
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
        result_data.append(float(str(f"{loss.item():3f}")))

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
        choices=["mnist", "fashion-mnist", "cifar10", "cifar100", "super-tiny-imagenet", "tiny-imagenet"],
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
    parser.add_argument(
        "--result-strategy",
        type=str,
        choices=["latent", "all"],
        help="If saving results, the independent variable to vary when saving results"
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["reconstruction", "anomaly-detection"],
        required=True,
        help="The task to perform (either 'reconstruction' or 'anomaly-detection')",
        default="reconstruction"
    )

    # Parse the arguments
    args = parser.parse_args()
    dataset = args.dataset
    device = args.device
    num_images = args.num_images
    generate_imgs = args.generate_imgs
    save_results = args.save_results
    task = args.task
    result_strategy = args.result_strategy

    return {
        "dataset": dataset,
        "device": device,
        "num_images": num_images,
        "generate_imgs": generate_imgs,
        "save_results": save_results,
        "result_strategy": result_strategy,
        "task": task
    }

def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    logging.basicConfig(level=logging.INFO)

    # Get the arguments
    global device
    config = get_args()
    device = config["device"]

    train_loader, test_loader, input_nodes, hidden_nodes = load_and_split_data(
        config["dataset"],
        config["task"]
    )

    model = elmae_init(input_nodes, hidden_nodes)

    train_model(model, train_loader)
    test_model(
        model, 
        test_loader, 
        config["dataset"], 
        config["generate_imgs"],
        config["num_images"],
        config["task"]
    )

    if config["save_results"]:
        result_strat = config["result_strategy"]
        dataset = config["dataset"]
        task = config["task"]

        if result_strat in ["latent", "all"]:
            # Save the latent representation
            plot_latent_representation(
                model,
                test_loader,
                dataset,
                task,
                f"elmae/plots/latents/{dataset}-latent-representation-{task}.png"
            )
        if result_strat == "all":
            # Save the hyperparameter performance
            result_file=f"elmae/data/{dataset}_{task}_performance.csv"
            save_result_data(
                result_data,
                result_file
            )

if __name__ == "__main__":
    main()
