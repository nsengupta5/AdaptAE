"""
File: util.py
Author: Nikhil Sengupta
Created on: November 6, 2023
Last Modified: December 12, 2023
Email: ns214@st-andrews.ac.uk

Description: 
    This file contains utility functions common to all models,
    such as assertions, visualizations of data, etc.

License:
    This code is released under the MIT License
"""

import logging
import matplotlib.pyplot as plt
import torch
from torch import nn
import csv

"""
Visualize the original and reconstructed images
:param originals: The original images
:type originals: torch.Tensor
:param reconstructions: The reconstructed images
:type reconstructions: torch.Tensor
:param dataset: The dataset used
:type dataset: str
:param n: The number of images to visualize
:type n: int
"""
def visualize_comparisons(originals, reconstructions, dataset, num_imgs, results_file):
    logging.info(f"Generating {num_imgs} images...")
    plt.figure(figsize=(20, 4))
    for i in range(num_imgs): # Display original images
        ax = plt.subplot(2, num_imgs, i + 1)
        if dataset in ["mnist", "fashion-mnist"]:
            plt.imshow(originals[i].reshape(28, 28))
        elif dataset in ["cifar10", "cifar100", "super-tiny-imagenet"]:
            plt.imshow(originals[i].reshape(3, 32, 32).transpose(1, 2, 0))
        else:
            plt.imshow(originals[i].reshape(3, 64, 64).transpose(1, 2, 0))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Display reconstructed images
        ax = plt.subplot(2, num_imgs, i + 1 + num_imgs)
        if dataset in ["mnist", "fashion-mnist"]:
            plt.imshow(reconstructions[i].reshape(28, 28))
        elif dataset in ["cifar10", "cifar100", "super-tiny-imagenet"]:
            plt.imshow(reconstructions[i].reshape(3, 32, 32).transpose(1, 2, 0))
        else:
            plt.imshow(reconstructions[i].reshape(3, 64, 64).transpose(1, 2, 0))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    # Save the images
    logging.info(f"Saving images to output file...")
    plt.savefig(results_file)

"""
Evaluate the network based on the test data and the predicted data
:param test_data: The test data
:type test_data: torch.Tensor
:param pred_data: The predicted data
:type pred_data: torch.Tensor
:return: The loss and accuracy
:rtype loss: torch.Tensor
:rtype accuracy: torch.Tensor
"""
def evaluate(model, test_data, pred_data):
    criterion = nn.MSELoss()
    loss = criterion(test_data, pred_data)
    accuracy = 0
    return loss, accuracy

"""
Save the results to a CSV file
:param dataset: The dataset used
:type dataset: str
:param phased: Boolean indicating whether the model was monitored in a phased manner
:type phased: bool
:param result_strategy: The result strategy used
:type result_strategy: str
"""
def save_result_data(model, dataset, phased, result_strategy, result_data):
    target_dir = "phased" if phased else "total"
    with open (f'{model}/data/{target_dir}/{result_strategy}_{dataset}_performance.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(result_data)

"""
Print the header of a stage
:param header: The header to print
:type header: str
"""
def print_header(header):
    result_str = "\n" + header + "\n" + "=" * len(header)
    print(result_str)

"""
Exit the program with an error message of the correct usage
:param msg: The error message to display
:type msg: str
:param parser: The parser to use to print the correct usage
:type parser: argparse.ArgumentParser
"""
def exit_with_error(msg, parser):
    logging.error(msg)
    parser.print_help()
    exit(1)

"""
Assert a condition and log the error if the condition is not met
:param condition: The condition to assert
:type condition: bool
:param msg: The error message to display
:type msg: str
"""
def assert_cond(condition, msg):
    try:
        assert condition, msg
    except AssertionError as e:
        logging.error(e)
        raise e
