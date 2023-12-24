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

import torch
from torchvision import datasets, transforms
import pandas as pd
import numpy as np
import random
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
        case 'mnist-corrupted':
            corrupt_data(dataset)
            train_data = pd.read_csv('./data/MNIST_CSV/mnist_train.csv')
            test_data = pd.read_csv('./data/MNIST_CSV/mnist_test_corrupted.csv')
            input_nodes = 784
            hidden_nodes = 128
        case 'fashion-mnist-corrupted':
            corrupt_data(dataset)
            train_data = pd.read_csv('./data/FashionMNIST_CSV/fashion-mnist_train.csv')
            test_data = pd.read_csv('./data/FashionMNIST_CSV/fashion-mnist_test_corrupted.csv')
            input_nodes = 784
            hidden_nodes = 128
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

def corrupt_data(dataset):
    parent_dir = ''
    if dataset == 'mnist-corrupted':
        dataset = 'mnist'
        parent_dir = './data/MNIST_CSV'
    elif dataset == 'fashion-mnist-corrupted':
        dataset = 'fashion-mnist'
        parent_dir = './data/FashionMNIST_CSV'

    if not os.path.exists('./data'):
        os.mkdir('./data')
    if not os.path.exists(parent_dir):
        os.mkdir(parent_dir)
    if not os.path.exists(f'{parent_dir}/{dataset}_test_corrupted.csv'):
        df = pd.read_csv(f'{parent_dir}/{dataset}_test.csv')
        anom = df[:1000]
        clean = df[1000:]
        for i in range(len(anom)):
            row = anom.iloc[i]
            for j in range(len(row) - 1):
                row[j+1] = min(255, row[j+1] + random.randint(100, 200))

        anom['label'] = 1
        clean['label'] = 0

        new_test = pd.concat([anom, clean])
        new_test.sample(frac=1)
        new_test.to_csv(f'{parent_dir}/{dataset}_test_corrupted.csv', index=False)

class MyLoader(torch.utils.data.Dataset):
    def __init__(self, dataset=None):
        super(MyLoader, self).__init__()
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset.iloc[idx]
        row = row.drop(labels={'label'})
        data = torch.from_numpy(np.array(row)/255).float()
        return data
    
class Loader(MyLoader):
    def __init__(self, dataset):
        super(Loader, self).__init__(dataset)
