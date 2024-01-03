"""
File: data.py
Author: Nikhil Sengupta
Created on: November 6, 2023
Last Modified: January 5, 2024
Email: ns214@st-andrews.ac.uk

Description: 
    This file contains a function that loads the data for a model
    and returns the input and hidden nodes as well as the training
    and test data

License:
    This code is released under the MIT License
"""

import torch
from torchvision import datasets, transforms
import numpy as np
import os

"""
Load the data and return the input and hidden nodes
:param dataset: The dataset to load
:type dataset: str
:return input_nodes: The number of input nodes
:rtype input_nodes: int
:return hidden_nodes: The number of hidden nodes
:rtype hidden_nodes: int
:return train_data: The training data
:rtype train_data: torch.utils.data.Dataset
:return test_data: The test data
:rtype test_data: torch.utils.data.Dataset
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
            check_tiny_imagenet()
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
            check_tiny_imagenet()
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

"""
Check if Tiny ImageNet dataset exists, if not download it
"""
def check_tiny_imagenet():
    if not os.path.exists('./data'):
        os.mkdir('./data')
    if not os.path.exists('./data/tiny-imagenet-200'):
        print("Downloading Tiny ImageNet dataset")
        os.system('wget http://cs231n.stanford.edu/tiny-imagenet-200.zip')
        os.system('unzip tiny-imagenet-200.zip')
        os.system('rm tiny-imagenet-200.zip')
        os.system('mv tiny-imagenet-200 ./data')

def add_noise(img):
    mean = 0.0
    std = 0.1
    sigma = std**0.4
    gauss = np.random.normal(mean, sigma, img.shape)
    noisy_img = img + gauss
    return np.clip(noisy_img, 0, 1)

class NoisyLoader(torch.utils.data.Dataset):
    def __init__(self, dataset):
        super(NoisyLoader, self).__init__()
        self.dataset = dataset
        self.num_noisy_imgs = 1000

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]

        if idx < self.num_noisy_imgs:
            noisy_img = add_noise(img.numpy())
            return torch.from_numpy(noisy_img).float(), label
        else:
            return img, label
