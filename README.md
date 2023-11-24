## CS5199 Experiments

### Datasets

This project makes use of the following datasets:

- **MNIST**: 70,000 grayscale images of handwritten digits (0-9), each 28x28 pixels, used for basic image recognition tasks. Split into 60,000 training and 10,000 test images.

- **Fashion-MNIST**: 70,000 grayscale images of fashion items in 10 categories, each 28x28 pixels. It's divided into 60,000 training and 10,000 test images

- **CIFAR-10**: 60,000 32x32 color images in 10 classes, with 6,000 images per class. There are 50,000 training images and 10,000 test images

- **CIFAR-100**: 60,000 32x32 color images, but across 100 classes containing 600 images each

### Usage

To run a vanilla autoencoder:

`python autoencoder/train-autoencoder.py <dataset>`

where `dataset` is one of `['mnist', 'fashion-mnist', 'cifar10', 'cifar100', 'tiny-imagenet']`

To run ELM-AE:

`python elmae/train-elm-ae.py <dataset>`

where `dataset` is one of `['mnist', 'fashion-mnist', 'cifar10', 'cifar100', 'tiny-imagenet']`

To run OS-ELM:

`python oselm/train-os-elm.py <mode> {batch size} <dataset>`

where:

- `dataset` is one of `['mnist', 'fashion-mnist', 'cifar10', 'cifar100', 'tiny-imagenet']`

- `mode` is one of `['sample', 'batch']`

- `batch size` is the batch size you want if you select mode as `batch`
