"""
File: train-os-elm.py
Author: Nikhil Sengupta
Created on: November 6, 2023
Last Modified: December 5, 2020
Email: ns214@st-andrews.ac.uk

Description: 
    This file contains my implementation of the Online Sequential Extreme Learning Machine (OS-ELM) algorithm. It monitors the performance of the model through metrics such as training time,
    peak memory usage and loss. It also generates images of the reconstructions and saves the results of the metrics to a CSV file for further analysis.

License:
    This code is released under the MIT License

Usage:
    python train-os-elm.py [-h] --mode {sample,batch} --dataset {mnist,fashion-mnist,cifar10,cifar100,super-tiny-imagenet,tiny-imagenet} [--batch-size BATCH_SIZE] [--device {cpu,mps,cuda}] 
                           [--seq-prop SEQ_PROP] [--generate-imgs] [--save-results] [--phased] [--result-strategy {batch-size,seq-prop,total}] [--num-images NUM_IMAGES]

    options:
      -h, --help            show the help message and exit
      --mode {sample,batch}
                            The mode of sequential training (either 'sample' or 'batch')
      --dataset {mnist,fashion-mnist,cifar10,cifar100,super-tiny-imagenet,tiny-imagenet}
                            The dataset to use (either 'mnist', 'fashion-mnist', 'cifar10', 'cifar100', 'super-tiny-imagenet' or 'tiny-imagenet')
      --batch-size BATCH_SIZE
                            The batch size to use. Defaults to 10 if not provided
      --device {cpu,mps,cuda}
                            The device to use (either 'cpu', 'mps' or 'cuda'). Defaults to 'cuda' if not provided
      --seq-prop SEQ_PROP   The sequential training data proportion. Must be between 0.01 and 0.99 inclusive. Defaults to 0.99 if not provided
      --generate-imgs       Whether to generate images of the reconstructions
      --save-results        Whether to save the results to a CSV file
      --phased              Whether to monitor and save phased or total performance results
      --result-strategy {batch-size,seq-prop,total}
                            If saving results, the independent variable to vary when saving results
      --num-images NUM_IMAGES
                            The number of images to generate. Defaults to 5 if not provided

Example: python train-os-elm.py --mode sample --dataset mnist 
"""

from oselm import OSELM
import torch
from torchvision import datasets, transforms
from torch.utils.data import random_split
import logging
import time
import warnings
import matplotlib.pyplot as plt
import psutil
import csv
import argparse

# Constants
DEFAULT_BATCH_SIZE = 10
DEFAULT_SEQ_PROP = 0.99
DEFAULT_NUM_IMAGES = 5
result_data = []

"""
Initialize the OSELM model
"""
def oselm_init(input_nodes, hidden_nodes):
    logging.info(f"Initializing OSELM model...")
    activation_func = 'sigmoid'
    loss_func = 'mse'
    logging.info(f"Initializing OSELM model complete.\n")
    return OSELM(activation_func, loss_func, input_nodes, hidden_nodes, device).to(device)

"""
Load and split the data into training, sequential and test data
param dataset: The dataset to load
param mode: The mode to load the data in
param batch_size: The batch size to use
param seq_prop: The proportion of the data to use for sequential training
return train_loader: The training data loader
return seq_loader: The sequential training data loader
return test_loader: The test data loader
return input_nodes: The number of input nodes
return hidden_nodes: The number of hidden nodes
"""
def load_and_split_data(dataset, mode, batch_size, seq_prop):
    logging.info(f"Loading and preparing data...")
    transform = transforms.ToTensor()
    input_nodes = 784
    hidden_nodes = 128

    # Set the batch size to 1 if in sample mode
    if mode == "sample":
        batch_size = 1

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
                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
            ])
            train_data = datasets.CIFAR10(root = './data', train = True, download = True, transform = transform)
            test_data = datasets.CIFAR10(root = './data', train = False, download = True, transform = transform)
            input_nodes = 3072
            hidden_nodes = 1024
        case 'cifar100':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
            ])
            train_data = datasets.CIFAR100(root = './data', train = True, download = True, transform = transform)
            test_data = datasets.CIFAR100(root = './data', train = False, download = True, transform = transform)
            input_nodes = 3072
            hidden_nodes = 1024
        case 'super-tiny-imagenet':
            transform = transforms.Compose([
                # Resize the images to 32x32 for smaller network
                transforms.Resize((32,32)),
                transforms.ToTensor(),
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
                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
            ])
            train_data = datasets.ImageFolder(root = './data/tiny-imagenet-200/train', transform = transform)
            test_data = datasets.ImageFolder(root = './data/tiny-imagenet-200/test', transform = transform)
            input_nodes = 12288
            hidden_nodes = 4096
        case _:
            raise ValueError(f"Invalid dataset: {dataset}")

    # Split the training data into training and sequential data
    seq_size = int(seq_prop * len(train_data))
    train_size = len(train_data) - seq_size
    train_data, seq_data = random_split(train_data, [train_size, seq_size])

    # Create the data loaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size = train_size, shuffle = True)
    seq_loader = torch.utils.data.DataLoader(seq_data, batch_size = batch_size, shuffle = True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size, shuffle = False)

    logging.info(f"Loading and preparing data complete.")
    return train_loader, seq_loader, test_loader, input_nodes, hidden_nodes

"""
Initialize the OSELM model with the initial training data
:param model: The OSELM model
:param train_loader: The initial training loader
:param phased: Boolean indicating if we're monitoring phased training
"""
def train_init(model, train_loader, phased):
    peak_memory = 0
    process = None

    for (data, _) in train_loader:
        # Reshape the data to fit the model
        data = data.reshape(-1, model.input_shape[0]).float().to(device)
        logging.info(f"Initial training on {len(data)} samples...")

        # Don't reset the peak memory if we're monitoring total memory
        if phased:
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            else:
                process = psutil.Process()
                peak_memory = process.memory_info().rss 

        start_time = time.time()

        model.init_phase(data)

        end_time = time.time()

        if phased:
            if device == "cuda":
                peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
            else:
                current_memory = process.memory_info().rss
                peak_memory = max(peak_memory, current_memory) / (1024 ** 2)

        # Evaluate the model on the initial training data
        pred = model.predict(data)
        loss, _ = model.evaluate(data, pred)

        # Print results
        print_header("Initial Training Benchmarks")
        if phased:
            print(f"Peak memory allocated during training: {peak_memory:.2f} MB")

        training_time = end_time - start_time
        print(f"Time taken: {training_time:.2f} seconds.")
        print(f"Initial training loss: {loss:.3f}")

        # Saving results
        if phased:
            result_data.append(training_time)
            result_data.append(round(peak_memory, 2))
            result_data.append(float(str(f"{loss:.3f}")))

        logging.info(f"Initial training complete\n")

"""
Train the OSELM model sequentially on the sequential training data
:param model: The OSELM model
:param seq_loader: The sequential training loader
:param mode: The mode of sequential training, either "sample" or "batch"
:param phased: Boolean indicating if we're monitoring phased training
"""
def train_sequential(model, seq_loader, mode, phased):
    logging.info(f"Sequential training on {len(seq_loader)} batches in {mode} mode...")

    # Metrics for each iteration
    total_loss = 0
    peak_memory = 0
    process = None

    # Don't reset the peak memory if we're monitoring total memory
    if phased:
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        else:
            process = psutil.Process()
            peak_memory = process.memory_info().rss

    start_time = time.time()
    for (data, _) in seq_loader:
        # Reshape the data to fit the model
        data = data.reshape(-1, model.input_shape[0]).float().to(device)

        model.seq_phase(data, mode)

        # Set peak memory to the max of the current memory and the peak memory if using CPU
        if phased:
            if device == "cpu":
                current_memory = process.memory_info().rss
                peak_memory = max(peak_memory, current_memory)

        pred = model.predict(data)
        loss, _ = model.evaluate(data, pred)
        total_loss += loss.item()
    
    end_time = time.time()

    if phased:
        if device == "cuda":
            peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)


    # Print results
    print_header("Sequential Training Benchmarks")
    if phased:
        print(f"Peak memory allocated during training: {peak_memory:.2f} MB")
    training_time = end_time - start_time
    print(f"Time taken: {training_time:.2f} seconds.")
    print(f"Average loss per batch: {total_loss / len(seq_loader):.2f}")

    # Saving results
    if phased:
        result_data.append(training_time)
        result_data.append(round(peak_memory, 2))
        result_data.append(float(str(f"{(total_loss / len(seq_loader)):3f}")))

    logging.info(f"Sequential training complete")

"""
Train the model
:param model: The model to train
:param train_loader: The training data loader
:param seq_loader: The sequential training data loader
:param mode: The mode of sequential training, either "sample" or "batch"
:param device: The device to use
:param phased: Boolean indicating if we're monitoring phased training
"""
def train_model(model, train_loader, seq_loader, mode, device, phased):
    peak_memory = 0
    process = None

    # Reset the peak memory if we're not monitoring phased training
    if not phased:
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        else:
            process = psutil.Process()
            peak_memory = process.memory_info().rss

    start_time = time.time()

    train_init(model, train_loader, phased)
    train_sequential(model, seq_loader, mode, phased)

    end_time = time.time()

    if not phased:
        if device == "cuda":
            peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
        else:
            current_memory = process.memory_info().rss
            peak_memory = max(peak_memory, current_memory) / (1024 ** 2)

    # Print results
    print_header("Total Training Benchmarks")
    training_time = end_time - start_time
    if not phased:
        print(f"Peak memory allocated during training: {peak_memory:.2f} MB")
        result_data.append(training_time)
        result_data.append(round(peak_memory, 2))
    print(f"Time taken: {training_time:.2f} seconds.")

    logging.info(f"Total training complete\n")

"""
Test the OSELM model on the test data
:param model: The OSELM model
:param test_data: The test data
"""
def test_model(model, test_loader, dataset, gen_imgs, num_imgs):
    logging.info(f"Testing on {len(test_loader.dataset)} batches...")

    losses = []
    outputs = []
    saved_img = False

    for (data, _) in test_loader:
        # Reshape the data to fit the model
        data = data.reshape(-1, model.input_shape[0]).float().to(device)

        # Predict and evaluate the model
        pred = model.predict(data)
        loss, _ = model.evaluate(data, pred)
        losses.append(loss.item())

        # If the batch size is less than the number of images we want to generate, save the outputs
        # So we can use multiple batches to generate the desired number of images
        if test_loader.batch_size < num_imgs:
            outputs.append((data, pred))
        if gen_imgs:
            if not saved_img:
                # Only save the first num_imgs images
                if test_loader.batch_size < num_imgs:
                    if len(outputs) > num_imgs:
                        full_data = torch.cat([data for (data, _) in outputs], dim=0)
                        full_pred = torch.cat([pred for (_, pred) in outputs], dim=0)
                        visualize_comparisons(
                            full_data.cpu().numpy(), 
                            full_pred.cpu().detach().numpy(), 
                            dataset, 
                            test_loader.batch_size,
                            num_imgs
                        )
                        saved_img = True
                else:
                    visualize_comparisons(
                        data.cpu().numpy(),
                        pred.cpu().detach().numpy(),
                        dataset,
                        test_loader.batch_size,
                        num_imgs
                    )
                    saved_img = True

    # Print results
    print_header("Testing Benchmarks")
    loss = sum(losses) / len(losses)
    print(f"Total Loss: {loss:.5f}\n")

    # Saving results
    result_data.append(float(str(f"{loss:.5f}")))

    logging.info(f"Testing complete.")

"""
Visualize the original and reconstructed images
:param originals: The original images
:param reconstructions: The reconstructed images
:param dataset: The dataset used
:param n: The number of images to visualize
"""
def visualize_comparisons(originals, reconstructions, dataset, batch_size, n):
    logging.info(f"Generating {n} images...")
    plt.figure(figsize=(20, 4))
    for i in range(n): # Display original images
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

    # Save the images
    logging.info(f"Saving images to oselm/results/ ...")
    if batch_size == 1:
        plt.savefig(f"oselm/results/{dataset}-reconstructions-sample.png")
    else:
        plt.savefig(f"oselm/results/{dataset}-reconstructions-batch-{batch_size}.png")

"""
Save the results to a CSV file
:param dataset: The dataset used
:param phased: Boolean indicating whether the model was monitored in a phased manner
:param result_strategy: The result strategy used
"""
def save_result_data(dataset, phased, result_strategy):
    target_dir = "phased" if phased else "total"
    with open (f'oselm/data/{target_dir}/{result_strategy}_{dataset}_performance.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(result_data)

""" 
Get the arguments from the command line
"""
def get_args():
    parser = argparse.ArgumentParser(description="Training an OS-ELM model")
    # Define the arguments
    parser.add_argument(
        "--mode", 
        type=str, 
        choices=["sample", "batch"],
        required=True,
        help="The mode of sequential training (either 'sample' or 'batch')"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["mnist", "fashion-mnist", "cifar10", "cifar100", "super-tiny-imagenet", "tiny-imagenet"],
        required=True,
        help="The dataset to use (either 'mnist', 'fashion-mnist', 'cifar10', 'cifar100', 'super-tiny-imagenet' or 'tiny-imagenet')"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="The batch size to use. Defaults to 10 if not provided"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "mps", "cuda"],
        default="cuda",
        help="The device to use (either 'cpu', 'mps' or 'cuda'). Defaults to 'cuda' if not provided"
    )
    parser.add_argument(
        "--seq-prop",
        type=float,
        default=DEFAULT_SEQ_PROP,
        help="The sequential training data proportion. Must be between 0.01 and 0.99 inclusive. Defaults to 0.99 if not provided"
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
        "--phased",
        action="store_true",
        help="Whether to monitor and save phased or total performance results"
    )
    parser.add_argument(
        "--result-strategy",
        type=str,
        choices=["batch-size", "seq-prop", "total"],
        help="If saving results, the independent variable to vary when saving results"
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=DEFAULT_NUM_IMAGES,
        help="The number of images to generate. Defaults to 5 if not provided"
    )

    # Parse the arguments
    args = parser.parse_args()
    mode = args.mode
    dataset = args.dataset
    device = args.device
    gen_imgs = args.generate_imgs
    save_results = args.save_results
    phased = args.phased
    result_strategy = args.result_strategy
    num_images = args.num_images

    # Assume sample mode if no mode is specified
    batch_size = 1
    if args.mode == "batch": 
        if args.batch_size is None:
            batch_size = DEFAULT_BATCH_SIZE
        else:
            batch_size = args.batch_size
    else:
        if args.batch_size is not None:
            # Batch size is not used in sample mode
            exit_with_error("Batch size is not used in sample mode", parser)

    seq_prop = DEFAULT_SEQ_PROP
    if args.seq_prop is not None:
        if args.seq_prop <= 0 or args.seq_prop >= 1:
            # Sequential proportion must be between 0 and 1
            exit_with_error("Sequential proportion must be between 0 and 1", parser)
        else:
            seq_prop = args.seq_prop

    if args.save_results:
        if args.result_strategy is None:
            # Must specify a result strategy if saving result
            exit_with_error("Must specify a result strategy if saving results", parser)

    return mode, dataset, batch_size, device, seq_prop, gen_imgs, save_results, phased, result_strategy, num_images

"""
Print the header of a stage
:param header: The header to print
"""
def print_header(header):
    result_str = "\n" + header + "\n" + "=" * len(header)
    print(result_str)


"""
Exit the program with an error message of the correct usage
:param msg: The error message to display
:param parser: The parser to use to print the correct usage
"""
def exit_with_error(msg, parser):
    logging.error(msg)
    parser.print_help()
    exit(1)


def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    logging.basicConfig(level=logging.INFO)

    # Get the arguments
    global device
    mode, dataset, batch_size, device, seq_prop, gen_imgs, save_results, phased, result_strategy, num_imgs = get_args()

    # Append independent variables to result data
    if save_results:
        match result_strategy:
            case "batch-size":
                result_data.append(batch_size)
            case "seq-prop":
                result_data.append(seq_prop)
            case "total":
                result_data.append(batch_size)
                result_data.append(seq_prop)

    train_loader, seq_loader, test_loader, input_nodes, hidden_nodes = load_and_split_data(dataset, mode, batch_size, seq_prop)
    model = oselm_init(input_nodes, hidden_nodes)
    train_model(model, train_loader, seq_loader, mode, device, phased)
    test_model(model, test_loader, dataset, gen_imgs, num_imgs)

    if save_results:
        save_result_data(dataset, phased, result_strategy)

if __name__ == "__main__":
    main()
