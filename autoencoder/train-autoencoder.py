"""
File: train-autoencoder.py
Author: Nikhil Sengupta
Created on: November 6, 2023
Last Modified: January 5, 2024
Email: ns214@st-andrews.ac.uk

Description: 
    This file contains the implementation of the training of the autoencoder model

License:
    This code is released under the MIT License

Usage:
    python train-autoencoder.py [-h] --dataset {mnist,fashion-mnist,cifar10,cifar100,
                                super-tiny-imagenet,tiny-imagenet} 
                                [--device {cpu,mps,cuda}] [--generate-imgs] 
                                [--save-results] [--num-images NUM_IMAGES]
                                [--num-epochs NUM_EPOCHS] [--batch-size BATCH_SIZE] 
                                [--result-strategy {batch-size,num-epochs,all-hyper,latent,all}] 
                                --task {reconstruction,anomaly-detection}

    Train an autoencoder model

    options:
      -h, --help            show this help message and exit

      --dataset {mnist,fashion-mnist,cifar10,cifar100,super-tiny-imagenet,tiny-imagenet}
                            The dataset to use 
                            (either 'mnist', 'fashion-mnist', 'cifar10', 
                            'cifar100', 'super-tiny-imagenet' or 'tiny-imagenet')

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

      --result-strategy {batch-size,num-epochs,all-hyper,latent,all}
                            If saving results, the independent variable to vary when saving results

      --task {reconstruction,anomaly-detection}
                            The task to perform (either 'reconstruction' or 'anomaly-detection')

Example: python train-autoencoder.py --dataset mnist --num-epochs 50 --task reconstruction
"""

from models.autoencoder import Autoencoder
from util.util import *
from util.data import *
import torch
from torch.utils.data import random_split
import logging
import time
import psutil
import warnings
import argparse
from skorch import NeuralNetRegressor
from sklearn.model_selection import GridSearchCV

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
    logging.info("Initialising Autoencoder model complete.\n")
    return model

"""
Load and split the data
:param dataset: The dataset to load
:type dataset: str
:param batch_size: The batch size
:type batch_size: int
:param task: The task to perform
:type task: str
:return train_loader: The training data loader
:rtype train_loader: torch.utils.data.DataLoader
:return valid_loader: The validation data loader
:rtype valid_loader: torch.utils.data.DataLoader
:return test_loader: The test data loader
:rtype test_loader: torch.utils.data.DataLoader
:return input_nodes: The number of input nodes
:rtype input_nodes: int
:return hidden_nodes: The number of hidden nodes
:rtype hidden_nodes: int
"""
def load_and_split_data(dataset, batch_size, task):
    logging.info(f"Loading and preparing data...")

    # Load the data
    input_nodes, hidden_nodes, train_data, test_data = load_data(dataset)

    train_size = int(0.8 * len(train_data))  
    valid_size = len(train_data) - train_size  

    train_data, valid_data = random_split(train_data, [train_size, valid_size])

    # Create the data loaders
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size = batch_size, shuffle = True)

    valid_loader = torch.utils.data.DataLoader(dataset=valid_data, batch_size = batch_size, shuffle = True)

    if task == "reconstruction":
        test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size = batch_size, shuffle = False)
    else:
        # Create the noisy data loader
        test_loader = torch.utils.data.DataLoader(
            NoisyLoader(test_data),
            batch_size=batch_size,
            shuffle=False
        )
    
    logging.info(f"Loading and preparing data complete.")
    return train_loader, valid_loader, test_loader, input_nodes, hidden_nodes

"""
Train the autoencoder model
:param model: The autoencoder model
:type model: Autoencoder
:param data_loader: The training data loader
:type data_loader: torch.utils.data.DataLoader
:param num_epochs: The number of epochs to train for
:type num_epochs: int
"""
def train_model(model, train_loader, valid_loader, num_epochs):
    logging.info(f"Training the autoencoder model...")
    
    # Set the model to training mode
    model.train()

    losses = []
    times = []

    # Time and CUDA memory tracking
    peak_memory = 0
    process = None
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    elif device == 'cpu':
        process = psutil.Process()

    start_time = time.time()

    # Find the best parameters
    logging.info(f"Finding the best parameters...")
    best_lr, best_weight_decay = find_best_params(model.input_shape[0], model.hidden_shape[0], valid_loader)
    optimizer = torch.optim.Adam(model.parameters(), lr=best_lr, weight_decay=best_weight_decay)
    logging.info(f"Finding the best parameters complete.\n")

    logging.info(f"Training on {len(train_loader)} batches...")
    for epoch in range(num_epochs):
        loss = 0
        epoch_start_time = time.time()
        for (img, _) in train_loader:
            # Reshape the image to fit the model
            img = img.reshape(-1, model.input_shape[0]).to(device)
            recon = model(img)
            train_loss, _ = evaluate(img, recon)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            loss += train_loss.item()

        epoch_end_time = time.time()
        loss /= len(train_loader)

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
    peak_memory_allocated = peak_memory / (1024 ** 2)
    avg_loss = np.median(losses)
    print(f"Peak memory allocated during training: {peak_memory_allocated:.2f} MB")
    print(f"Time taken: {training_time:.2f} seconds.")
    print(f"Average loss per epoch: {avg_loss:.5f}")
    result_data.append(training_time)
    result_data.append(round(peak_memory_allocated, 2))
    result_data.append(float(str(f"{avg_loss:.3f}")))

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
:param task: The task to perform
:type task: str
"""
def test_model(model, data_loader, dataset, gen_imgs, num_imgs, task, batch_size):
    logging.info(f"Testing the autoencoder model...")

    # Set the model to evaluation mode
    model.eval()

    losses = []
    logging.info(f"Testing on {len(data_loader)} batches...")
    saved_img = False
    with torch.no_grad():
        for (img, _) in data_loader:
            # Reshape the image to fit the model
            img = img.reshape(-1, model.input_shape[0]).to(device)
            recon = model(img)
            loss, _ = evaluate(img, recon)
            losses.append(loss.item())
            if gen_imgs:
                # Only save the first num_imgs images
                if not saved_img:
                    results_file = (
                        f"autoencoder/results/reconstruction/{dataset}-{task}.png"
                        if task == "reconstruction"
                        else f"autoencoder/results/anomaly_detection/{dataset}-{task}.png"
                    )
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
    loss = sum(losses) / len(losses)
    print(f'Loss: {loss:.5f}\n')
    result_data.append(float(str(f"{loss:.3f}")))

    if task == "anomaly-detection":
        loss_file = f"autoencoder/plots/losses/{dataset}-anomaly-losses.png"
        confusion_file = f"autoencoder/plots/confusion/{dataset}-confusion-matrix.png"
        plot_loss_distribution(model.name, losses, dataset, batch_size, loss_file, confusion_file)

    logging.info(f"Testing complete.")

"""
Find the best parameters for the autoencoder model using GridSearchCV
:param input_nodes: The number of input nodes
:type input_nodes: int
:param hidden_nodes: The number of hidden nodes
:type hidden_nodes: int
:param valid_loader: The validation data loader
:type valid_loader: torch.utils.data.DataLoader
:returns: The best learning rate and weight decay
:rtype: float, float
"""
def find_best_params(input_nodes, hidden_nodes, valid_loader):
    all_features = []
    for batch_features, _ in valid_loader:
        # Flatten the batch features if necessary and convert to numpy array
        batch_features = batch_features.view(batch_features.size(0), -1).numpy()
        all_features.append(batch_features)

    # Concatenate all batches
    train_data = np.concatenate(all_features, axis=0)

    autoencoder_net = NeuralNetRegressor(
        module=Autoencoder,
        module__n_input_nodes=input_nodes,
        module__n_hidden_nodes=hidden_nodes,
        module__device=device,
        device=device,
        criterion=nn.MSELoss,
        max_epochs=10,
        lr=1e-1,
        batch_size=64,
        optimizer=torch.optim.Adam,
        optimizer__weight_decay=1e-5,
    )

    param_grid = {
        'lr': [1e-1, 1e-2, 1e-3],
        'optimizer__weight_decay': [1e-3, 1e-4, 1e-5]
    }

    # Set up GridSearchCV
    grid = GridSearchCV(autoencoder_net, param_grid, refit=False, cv=3, scoring='neg_mean_squared_error')
    grid.fit(train_data, train_data)

    # Print the best parameters
    print_header("Best Parameters")
    print(f"Best parameters: {grid.best_params_}")
    return grid.best_params_['lr'], grid.best_params_['optimizer__weight_decay']

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
:return task: The task to perform
:rtype task: str
"""
def get_args():
    parser = argparse.ArgumentParser(description='Train an autoencoder model')
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
    parser.add_argument(
        "--result-strategy",
        type=str,
        choices=["batch-size", "num-epochs", "all-hyper", "latent", "all"],
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
    generate_imgs = args.generate_imgs
    save_results = args.save_results
    num_images = args.num_images
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    result_strategy = args.result_strategy
    task = args.task

    if args.save_results:
        if args.result_strategy is None:
            # Must specify a result strategy if saving result
            exit_with_error("Must specify a result strategy if saving results", parser)

    return {
        "dataset": dataset,
        "device": device,
        "generate_imgs": generate_imgs,
        "save_results": save_results,
        "num_images": num_images,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "result_strategy": result_strategy,
        "task": task
    }

def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    logging.basicConfig(level=logging.INFO)

    # Get the arguments
    config = get_args()
    global device
    device = config["device"]

    # Append independent variable to result data
    if config["save_results"]:
        match config["result_strategy"]:
            case "batch-size":
                result_data.append(config["batch_size"])
            case "num-epochs":
                result_data.append(config["n_epochs"])
            case "all-hyper" | "all":
                result_data.append(config["batch_size"])
                result_data.append(config["num_epochs"])

    # Load the data
    train_loader, valid_loader, test_loader, input_nodes, hidden_nodes = load_and_split_data(
        config["dataset"], 
        config["batch_size"],
        config["task"]
    )

    model = autoencoder_init(input_nodes, hidden_nodes)

    train_model(model, train_loader, valid_loader, config["num_epochs"])
    test_model(
        model, 
        test_loader, 
        config["dataset"], 
        config["generate_imgs"], 
        config["num_images"], 
        config["task"],
        config["batch_size"]
    )

    if config["save_results"]:
        result_strat = config["result_strategy"]
        dataset = config["dataset"]
        task = config["task"]

        if result_strat in ["latent", "all"]:
            # Plot the latent representation
            plot_latent_representation(
                model, 
                test_loader, 
                dataset,
                task,
                f"autoencoder/plots/latents/{dataset}-latent-representation-{task}.png",
            )
        if result_strat in ["all-hyper", "all"]:
            # Plot the hyperparameter performance
            strat = "total" if result_strat in ["all", "all-hyper"] else result_strat
            save_result_data(
                result_data,
                f"autoencoder/data/{strat}_{dataset}_{task}_performance.csv"
            )

if __name__ == "__main__":
    main()
