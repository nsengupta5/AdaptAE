"""
File: data.py
Author: Nikhil Sengupta
Created on: November 6, 2023
Last Modified: December 12, 2023
Email: ns214@st-andrews.ac.uk

Description: 
    This file contains a function that loads the data for a model
    and returns the input and hidden nodes as well as the training
    and test data

License:
    This code is released under the MIT License
"""

from torchvision import datasets, transforms

"""
Load the data and return the input and hidden nodes
:param dataset: The dataset to load
:return input_nodes: The number of input nodes
:return hidden_nodes: The number of hidden nodes
:return train_data: The training data
:return test_data: The test data
"""
def load_data(dataset):
    transform = transforms.ToTensor()
    input_nodes = 784
    hidden_nodes = 128

    match dataset:
        case 'mnist':
            transform = transforms.ToTensor()
            train_data = datasets.MNIST(
                root = './data', 
                train = True, 
                download = True, 
                transform = transform
            )
            test_data = datasets.MNIST(
                root = './data',
                train = False,
                download = True,
                transform = transform
            )
        case 'fashion-mnist':
            transform = transforms.ToTensor()
            train_data = datasets.FashionMNIST(
                root = './data',
                train = True,
                download = True,
                transform = transform
            )
            test_data = datasets.FashionMNIST(
                root = './data',
                train = False,
                download = True,
                transform = transform
            )
        case 'cifar10':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
            ])
            train_data = datasets.CIFAR10(
                root = './data', 
                train = True, 
                download = True, 
                transform = transform
            )
            test_data = datasets.CIFAR10(
                root = './data', 
                train = False, 
                download = True, 
                transform = transform
            )
            input_nodes = 3072
            hidden_nodes = 1024
        case 'cifar100':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
            ])
            train_data = datasets.CIFAR100(
                root = './data', 
                train = True, 
                download = True, 
                transform = transform
            )
            test_data = datasets.CIFAR100(
                root = './data', 
                train = False, 
                download = True, 
                transform = transform
            )
            input_nodes = 3072
            hidden_nodes = 1024
        case 'super-tiny-imagenet':
            transform = transforms.Compose([
                # Resize the images to 32x32 for smaller network
                transforms.Resize((32,32)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
            ])
            train_data = datasets.ImageFolder(
                root = './data/tiny-imagenet-200/train', 
                transform = transform
            )
            test_data = datasets.ImageFolder(
                root = './data/tiny-imagenet-200/test', 
                transform = transform
            )
            input_nodes = 3072
            hidden_nodes = 1024
        case 'tiny-imagenet':
            transform = transforms.Compose([
                transforms.Resize((64,64)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
            ])
            train_data = datasets.ImageFolder(
                root = './data/tiny-imagenet-200/train', 
                transform = transform
            )
            test_data = datasets.ImageFolder(
                root = './data/tiny-imagenet-200/test', 
                transform = transform
            )
            input_nodes = 12288
            hidden_nodes = 4096
        case _:
            raise ValueError(f"Invalid dataset: {dataset}")

    return input_nodes, hidden_nodes, train_data, test_data
